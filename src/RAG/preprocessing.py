from pypdf import PdfReader
import re
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer

def optimize_chapter_extraction(extract_data):
    """
    Extracts data segments based on chapter markers in the second element of each tuple.

    Args:
        extract_data: A list of tuples, where the second element is a string containing potential chapter markers.

    Returns:
        A list of lists, where each sublist contains data segments between chapter markers.
    """

    ans = []
    start_index = 0

    for i in range(len(extract_data)):
        if re.search(r"Chương \d+", extract_data[i][1]):
            if i > 0:
                ans.append(extract_data[start_index:i])
            start_index = i

    ans.append(extract_data[start_index:]) 
    return ans

def format_data(ans):
    parsed_data = defaultdict(dict)
    for chapter_lines in ans:
        current_chapter = ""
        current_title = ""
        chapter_found = False

        for line in chapter_lines:
            line = line.strip()

            # Match chapter titles (e.g., 'Chương 1')
            if not chapter_found:
                chapter_match = re.match(r'^Chương (\d+)', line, re.IGNORECASE)
                if chapter_match:
                    current_chapter = f"Chương {chapter_match.group(1)}"
                    chapter_found = True
                    continue
            elif re.match(r'^Chương (\d+)', line, re.IGNORECASE):
                continue            

            # Match the title of the chapter (the next line or two after 'Chương')
            if current_chapter and not current_title and line:
                current_title = line
                parsed_data[current_chapter]['title'] = current_title
                continue

            # Match subheadings like '1.1.', '1.2.'
            subheading_match = re.match(r'^(\d+(?:\.\d+)*\.?)\s+(.*)', line)
            if subheading_match:
                subheading = subheading_match.group(1)
                subheading_title = subheading_match.group(2).strip()
                parsed_data[current_chapter].setdefault('subsections', {})[subheading] = {
                    'title': subheading_title,
                    'content': []
                }
                current_subheading = subheading
                continue

            # Add content to the last matched subheading
            if current_chapter and 'subsections' in parsed_data[current_chapter]:
                subsections = parsed_data[current_chapter]['subsections']
                if current_subheading in subsections:
                    subsections[current_subheading]['content'].append(line)

    # Clean up content by joining lines
    for chapter, data in parsed_data.items():
        if 'subsections' in data:
            for subheading, details in data['subsections'].items():
                details['content'] = " ".join([line for line in details['content'] if line])
    
    return parsed_data

def dataframe_to_dict(parsed_data):
    data_for_df = []

    for chapter, data in parsed_data.items():
        chapter_title = data['title']
        if 'subsections' in data:
            for subheading, details in data['subsections'].items():
                data_for_df.append({
                    "Chapter": chapter,
                    "Chapter Title": chapter_title,
                    "Subheading": subheading,
                    "Subheading Title": details['title'],
                    "Content": details['content']
                })
    df = pd.DataFrame(data_for_df)
    return df.to_dict("records")

def chunk_content(data, max_length=256):
    chunks = []
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    for item in data:
        content = item["Content"]
        tokens = tokenizer.tokenize(content)
        
        for i in range(0, len(tokens), max_length):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            
            # Lưu chunk cùng metadata
            chunks.append({
                "Tên bệnh": item["Chapter Title"],
                "Loại bệnh": item["Subheading Title"],
                "Chunk nội dung": chunk_text
            })
    return chunks

def preprocessing(link):
    reader = PdfReader(link)

    extract_data = []

    for page in reader.pages:
        extract_data.append(page.extract_text().split("\n"))
    data_ex = optimize_chapter_extraction(extract_data)

    ans = []
    for d in data_ex:
        da_l = []
        for e in d:
            da_l.extend(e)
        ans.append(da_l)

    parsed_data = format_data(ans)
    text_dict = dataframe_to_dict(parsed_data)

    return text_dict

if __name__ == "__main__":
    reader = PdfReader("../../public/document.pdf")

    extract_data = []

    for page in reader.pages:
        extract_data.append(page.extract_text().split("\n"))
    data_ex = optimize_chapter_extraction(extract_data)

    ans = []
    for d in data_ex:
        da_l = []
        for e in d:
            da_l.extend(e)
        ans.append(da_l)

    parsed_data = format_data(ans)
    text_dict = dataframe_to_dict(parsed_data)
    # print(text_dict)

