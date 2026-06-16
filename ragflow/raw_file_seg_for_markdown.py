from markdown import markdown 
from deepdoc.parser import MarkdownParser
from io import BytesIO
from PIL import Image
from rag.nlp import find_codec
import os


class Markdown(MarkdownParser):
    def get_picture_urls(self, sections):
        if not sections:
            return []
        if isinstance(sections, type("")):
            text = sections
        elif isinstance(sections[0], type("")):
            text = sections[0]
        else:
            return []
        
        from bs4 import BeautifulSoup
        html_content = markdown(text)
        soup = BeautifulSoup(html_content, 'html.parser')
        html_images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
        return html_images
    
    # def get_pictures(self, text, filepath):
    #     """Download and open all images from markdown text."""
    #     # import requests
    #     image_urls = self.get_picture_urls(text)
    #     # print(image_urls)
    #     images = []
    #     # Find all image URLs in text
    #     # for url in image_urls:
    #     #     try:
    #     #         response = requests.get(url, stream=True, timeout=30)
    #     #         if response.status_code == 200 and response.headers['Content-Type'].startswith('image/'):
    #     #             img = Image.open(BytesIO(response.content)).convert('RGB')
    #     #             images.append(img)
    #     #     except Exception as e:
    #     #         continue
        
    #     # 在本地资料夹中寻找相应的图片
    #     for url in image_urls:
    #         img_filepath = os.path.join(filepath, url)
    #         try:
    #             if os.path.isfile(img_filepath):
    #                 img = Image.open(img_filepath).convert('RGB')
    #                 images.append(img)
    #         except Exception as e:
    #             continue
                    
    #     return images if images else None
    
    def get_pictures(self, url, filepath):
        # 在本地资料夹中寻找相应的图片
        img_filepath = os.path.join(filepath, url)
        try:
            if os.path.isfile(img_filepath):
                img = Image.open(img_filepath).convert('RGB')
        except Exception as e:
            pass
                    
        return img if img else None

    def __call__(self, filename, binary=None):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(filename, "r", encoding="utf-8") as f:
                txt = f.read()
        remainder, tables = self.extract_tables_and_remainder(f'{txt}\n')
        sections = []
        tbls = []
        for sec in remainder.split("\n"):
            if sec.strip().find("#") == 0:
                sections.append((sec, ""))
            elif sections and sections[-1][0].strip().find("#") == 0:
                sec_, _ = sections.pop(-1)
                sections.append((sec_ + "\n" + sec, ""))
            else:
                sections.append((sec, ""))
        for table in tables:
            tbls.append(((None, markdown(table, extensions=['markdown.extensions.tables'])), ""))
        return sections, tbls

if __name__ == "__main__":

    filename = r"D:\Porject\RAG_with_deepseek\minerU_output\1\123_250804_011920.md"
    sections, tables = Markdown()(filename)
    # print(sections)
    # print(tables)
    # print(len(sections))
    for sec in sections:
        print(sec)
        print('\n')



