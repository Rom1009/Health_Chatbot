import requests
from bs4 import BeautifulSoup
import pandas as pd


articles = []

# URL của trang cần thu thập dữ liệu
for i in range(0,826,15):
    url = f"http://www.benhviennhi.org.vn/news/cat/77/{i}.html"
    response = requests.get(url)
    response.encoding = "utf-8"

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, "html.parser")

        # Danh sách để lưu thông tin

        divs = soup.find_all("div", style=lambda value: "clear:both; margin-top:10px; padding:2px 5px 10px 5px; overflow:hidden; border-bottom:1px solid #99CC00" 
                    in value if value else False)

        # Tìm các bài viết
        for article in divs:
            title_tag = article.find("a")
            if title_tag: 
                href = title_tag["href"]
                articles.append({"Title": title_tag.get_text(strip = True), "Link": href})
    else:
        print(f"Không thể truy cập trang chính: {url} (Mã lỗi: {response.status_code})")


data = []

for article in articles:
    print(f"Đang truy cập:{article["Link"]} ")

    try: 
        detail_response = requests.get(article["Link"])
        detail_response.encoding = 'utf-8'

        if detail_response.status_code == 200:
            detail_soup = BeautifulSoup(detail_response.text, "html.parser")

            types = detail_soup.find("div", class_ = "content_top").findAll("a")[1]
            Types = types.get_text(strip = True)
            body_divs = detail_soup.find("div", style = lambda value: "width: 98%; overflow:hidden" in value if value else False)
            
            titles = body_divs.find("h2").get_text(strip = True)

            text = body_divs.find("div", id = "read_news").get_text(separator="\n", strip = True)
            
            # Lưu thông tin vào file CSV
            data.append({
                "Type": Types,
                "Title": titles,
                "Text": text,
            })
        else:
            print(f"Không thể truy cập bài viết: {article['Link']} (Mã lỗi: {detail_response.status_code})")
            
    except Exception as e:
        print(f"Lỗi khi truy cập: {article['Link']} - {e}")
        
    print("-" * 50)

df = pd.DataFrame(data)
df.to_csv("benhviennhi_articles.csv", index=False, encoding="utf-8")
print("Dữ liệu đã lưu vào benhviennhi_articles.csv")