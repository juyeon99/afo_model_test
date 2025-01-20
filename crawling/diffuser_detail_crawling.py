# 디퓨저 정보 크롤링
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import mysql.connector
import re, time, os
from dotenv import load_dotenv

load_dotenv()

# MySQL 연결 설정
connection = mysql.connector.connect(
    host=os.getenv('MySQL_DB_HOST'),
    user=os.getenv('MySQL_DB_USER'),
    password=os.getenv('MySQL_DB_PASSWORD'),
    database=os.getenv('MySQL_DB_DATABASE'),
    charset='utf8mb4'
)

# Selenium 웹 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 20)  # 명시적 대기 시간 설정
base_url = 'https://www.bysuco.com/product?num=60&page='
page_number = 1

while True:
    try:
        # 각 페이지에 접근
        driver.get(f'{base_url}{page_number}&orderBy=popular&category_id%5B%5D=6&keyword=&kind=bt')
        time.sleep(2)  # 페이지 로딩 대기 시간 증가

        # 상품 목록에서 상세 페이지 링크 추출
        product_links = []
        elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[href^="/product/show"]')))
        
        if not elements:
            print("더 이상 상품이 없습니다. 수집을 종료합니다.")
            break

        # 링크 수집
        for element in elements:
            href = element.get_attribute('href')
            if href not in product_links:
                product_links.append(href)

        # 각 상품 상세 페이지 처리
        for link in product_links:
            try:
                driver.get(link)
                time.sleep(2)  # 페이지 로딩 대기

                # 기본 정보 추출
                try:
                    name_kr = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'desc.ellipsisTwo'))).text
                    
                    # 제외할 키워드 체크
                    if any(keyword in name_kr for keyword in ['벌크제품', '세트', '리필', '선물패키지', '스프레이']):
                        print(f"제외된 상품: {name_kr}")
                        continue  # 다음 상품으로 건너뜀
                        
                    print(f"상품명: {name_kr}")
                    
                    name_en = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'desc.enDesc.ellipsisTwo'))).text
                    print(f"상품명: {name_en}")
                    
                    brand_name = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'tit'))).text
                    print(f"브랜드: {brand_name}")
                    
                    desc_wrap = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'descWrap'))).text
                    print(f"설명: {desc_wrap}")
                    
                    ingredients_description = ""
                    
                    try:
                        notice_tables = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'noticeTable')))
    
                        if len(notice_tables) > 1:
                            second_notice_table = notice_tables[1]  # 두 번째 테이블 선택

                            rows = second_notice_table.find_elements(By.TAG_NAME, 'tr')

                            for row in rows:
                                th_elements = row.find_elements(By.TAG_NAME, 'th')
                                
                                # 각 <th>에 대해 "표시성분"을 확인
                                for th_element in th_elements:
                                    if th_element.text.strip() == "표시성분":
                                        # 해당 <th>에 맞는 <td>에서 ingredients description 추출
                                        td_element = row.find_element(By.TAG_NAME, 'td')
                                        ingredients_description = td_element.text.strip()
                                        print(f"표시성분 설명: {ingredients_description}")
                                        break
                        
                    except TimeoutException as e:
                        print(f"테이블 로딩 중 오류 발생: {e}")
                    except Exception as e:
                        print(f"기타 오류 발생: {e}")
                    
                    # 사이즈 옵션 추출 부분 수정
                    try:    
                        print("사이즈 옵션 추출 시작")
                        # 명시적 대기 추가
                        time.sleep(1)
                        
                        # 모든 optionBtn 찾기 (상태와 관계없이)
                        option_buttons = driver.find_elements(By.CSS_SELECTOR, 'button.optionBtn')
                        print(f"찾은 버튼 수: {len(option_buttons)}")
                        
                        size_options = []
                        for button in option_buttons:
                            try:
                                # i 태그의 ellipsisTwo 클래스를 가진 요소 찾기
                                size_text = button.find_element(By.CSS_SELECTOR, 'i.ellipsisTwo').text.strip()
                                if size_text:
                                    size_options.append(size_text)
                                    print(f"추출된 옵션: {size_text}")
                            except Exception as e:
                                print(f"버튼 처리 중 에러: {str(e)}")
                                continue
                        
                        size_options_str = ','.join(size_options) if size_options else 'No size options'
                        print(f"최종 사이즈 옵션: {size_options_str}")
                        
                    except Exception as e:
                        print(f"사이즈 옵션 추출 중 에러: {str(e)}")
                        size_options_str = ''
                    
                except TimeoutException as e:
                    print(f"기본 정보 추출 중 타임아웃: {e}")
                    continue

                # 이미지 처리
                image_urls = []
                try:
                    # 먼저 현재 표시된 이미지의 URL 가져오기
                    main_image = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.swiper-slide-active img'))
                    )
                    main_url = main_image.get_attribute('src')
                    if main_url:
                        image_urls.append(main_url)
                        print(f"메인 이미지 URL 추가: {main_url}")

                    # 모든 이미지 슬라이드 찾기
                    all_slides = driver.find_elements(By.CSS_SELECTOR, '.swiperWrap.mb .swiper-slide')
                    print(f"발견된 총 슬라이드 수: {len(all_slides)}")

                    # 각 슬라이드 순회
                    for i in range(len(all_slides)):
                        try:
                            # 현재 슬라이드의 이미지 찾기
                            current_slide = all_slides[i]
                            img = current_slide.find_element(By.TAG_NAME, 'img')
                            image_url = img.get_attribute('src')
                            
                            if image_url and image_url not in image_urls:
                                image_urls.append(image_url)
                                print(f"이미지 URL {i+1} 추가: {image_url}")

                        except Exception as e:
                            print(f"슬라이드 {i+1} 처리 중 오류: {e}")
                            continue

                except Exception as e:
                    print(f"이미지 처리 중 오류 발생: {e}")

                print(f"총 수집된 이미지 URL 수: {len(image_urls)}")
                
                # MySQL 데이터 삽입
                with connection.cursor() as cursor:
                    # 테이블 생성
                    cursor.execute('''CREATE TABLE IF NOT EXISTS diffuser (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        name_kr VARCHAR(50),
                        name_en VARCHAR(50),
                        brand VARCHAR(50),
                        description TEXT,
                        ingredients TEXT,
                        size_options TEXT
                    )''')
                    
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS diffuser_image (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        img_url TEXT,
                        diffuser_id BIGINT,
                        FOREIGN KEY (diffuser_id) REFERENCES diffuser(id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE
                    )
                    ''')

                    try:
                        # diffuser 테이블에 데이터 삽입
                        cursor.execute("""
                            INSERT INTO diffuser (name_kr, name_en, brand, description, ingredients, size_options) 
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (name_kr, name_en, brand_name, desc_wrap, ingredients_description, size_options_str))
                        diffuser_id = cursor.lastrowid
                        
                        # 이미지 URL 저장
                        for url in image_urls:
                            cursor.execute("""
                                INSERT INTO diffuser_image (img_url, diffuser_id) 
                                VALUES (%s, %s)
                            """, (url, diffuser_id))
                        connection.commit()
                        print(f"상품 '{name_kr}' 데이터 저장 완료")

                    except Exception as e:
                        connection.rollback()
                        print(f"데이터베이스 저장 중 오류: {e}")
                        continue

            except Exception as e:
                print(f"상품 페이지 처리 중 오류 발생: {e}")
                continue

        page_number += 1

    except Exception as e:
        print(f"페이지 처리 중 오류 발생: {e}")
        break

driver.quit()
connection.close()
print("크롤링이 완료되었습니다!")