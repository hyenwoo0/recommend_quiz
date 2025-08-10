import mysql.connector
from mysql.connector import Error
import os  # 환경 변수 사용을 위해 os 모듈을 임포트합니다.
from dotenv import load_dotenv # .env 파일 사용을 위해 라이브러리 임포트

# .env 파일에서 환경 변수를 로드합니다.
# 이 코드는 .env 파일에 정의된 변수들을 os.getenv로 읽을 수 있게 해줍니다.
load_dotenv()

def check_db_connection_from_env():
    """환경 변수를 사용하여 안전하게 MySQL 데이터베이스 연결을 확인하는 함수"""

    conn = None
    try:
        # 1. os.getenv()를 사용해 환경 변수에서 접속 정보를 읽어옵니다.
        db_host = os.getenv("DB_HOST", "127.0.0.1") # 값이 없으면 기본값 "127.0.0.1" 사용
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_DATABASE")

        # 2. 필수 환경 변수가 설정되었는지 확인합니다.
        if not all([db_user, db_password, db_name]):
            print("❌ 오류: 필요한 DB 접속 정보(DB_USER, DB_PASSWORD, DB_DATABASE)가 .env 파일에 설정되지 않았습니다.")
            return

        # 3. 읽어온 정보로 DB에 연결합니다.
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )

        if conn.is_connected():
            print("✅ 데이터베이스에 성공적으로 연결되었습니다. (.env 사용)")
            db_info = conn.get_server_info()
            print(f"   - MySQL 서버 버전: {db_info}")

    except Error as e:
        print(f"❌ 데이터베이스 연결에 실패했습니다: {e}")

    finally:
        if conn and conn.is_connected():
            conn.close()
            print("   - 데이터베이스 연결을 종료했습니다.")


if __name__ == "__main__":
    check_db_connection_from_env()
