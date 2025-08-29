import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

def insert_quiz_history():
    """
    사용자 이름과 퀴즈 코드를 문자열 그대로 데이터베이스에 삽입합니다.
    (주의: DB 테이블의 user_id, quiz_id 컬럼 타입이 VARCHAR여야 합니다.)
    """

    # 1. 데이터 정의
    history_data = [
        (1, 'user3', 'Q18', 0, 45), (2, 'user2', 'Q10', 1, 12),
        (3, 'user2', 'Q07', 1, 6), (4, 'user2', 'Q14', 1, 52),
        (5, 'user2', 'Q11', 1, 22), (6, 'user5', 'Q11', 0, 20),
        (7, 'user2', 'Q09', 1, 19), (8, 'user2', 'Q16', 1, 13),
        (9, 'user2', 'Q01', 0, 52), (10, 'user2', 'Q05', 0, 11),
        (11, 'user1', 'Q09', 0, 48), (12, 'user3', 'Q08', 0, 52),
        (13, 'user2', 'Q08', 1, 39), (14, 'user2', 'Q02', 1, 10),
        (15, 'user5', 'Q15', 1, 42), (16, 'user4', 'Q12', 1, 32),
        (17, 'user5', 'Q04', 1, 7), (18, 'user4', 'Q09', 1, 6),
        (19, 'user3', 'Q19', 1, 10), (20, 'user5', 'Q07', 1, 18),
        (21, 'user5', 'Q17', 0, 19), (22, 'user2', 'Q04', 1, 37),
        (23, 'user2', 'Q17', 0, 43), (24, 'user2', 'Q14', 0, 6),
        (25, 'user3', 'Q01', 1, 40), (26, 'user1', 'Q19', 0, 17),
        (27, 'user2', 'Q19', 1, 50), (28, 'user5', 'Q12', 0, 46),
        (29, 'user5', 'Q16', 1, 49), (30, 'user2', 'Q10', 0, 39),
        (31, 'user3', 'Q20', 1, 31), (32, 'user2', 'Q05', 0, 19),
        (33, 'user5', 'Q08', 0, 33), (34, 'user3', 'Q13', 1, 42),
        (35, 'user5', 'Q20', 1, 22), (36, 'user4', 'Q19', 0, 56),
        (37, 'user3', 'Q12', 0, 60), (38, 'user5', 'Q18', 1, 5),
        (39, 'user5', 'Q01', 0, 53), (40, 'user3', 'Q04', 1, 56),
        (41, 'user3', 'Q02', 1, 15), (42, 'user5', 'Q13', 0, 49),
        (43, 'user1', 'Q15', 0, 32), (44, 'user1', 'Q03', 1, 26),
        (45, 'user3', 'Q14', 1, 22), (46, 'user4', 'Q07', 0, 14),
        (47, 'user5', 'Q05', 1, 18), (48, 'user2', 'Q20', 1, 53),
        (49, 'user4', 'Q13', 0, 26), (50, 'user1', 'Q11', 1, 11),
        (51, 'user3', 'Q08', 1, 10), (52, 'user5', 'Q03', 1, 29),
        (53, 'user4', 'Q18', 1, 11), (54, 'user4', 'Q04', 1, 27),
        (55, 'user4', 'Q20', 1, 59), (56, 'user1', 'Q06', 0, 27),
        (57, 'user5', 'Q06', 0, 43), (58, 'user2', 'Q11', 0, 21),
        (59, 'user4', 'Q15', 1, 56), (60, 'user5', 'Q02', 1, 7),
        (61, 'user1', 'Q10', 1, 51), (62, 'user5', 'Q14', 0, 34),
        (63, 'user1', 'Q14', 1, 39), (64, 'user5', 'Q09', 1, 12),
        (65, 'user4', 'Q16', 1, 29), (66, 'user5', 'Q13', 0, 10),
        (67, 'user3', 'Q07', 1, 40), (68, 'user1', 'Q12', 1, 23),
        (69, 'user3', 'Q16', 0, 58), (70, 'user4', 'Q11', 1, 45),
        (71, 'user2', 'Q03', 1, 44), (72, 'user4', 'Q02', 1, 60),
        (73, 'user4', 'Q06', 0, 28), (74, 'user5', 'Q19', 1, 41),
        (75, 'user5', 'Q15', 0, 17), (76, 'user1', 'Q17', 1, 50),
        (77, 'user5', 'Q12', 1, 9), (78, 'user4', 'Q03', 1, 7),
        (79, 'user4', 'Q08', 0, 47), (80, 'user2', 'Q12', 1, 19),
        (81, 'user4', 'Q17', 1, 54), (82, 'user3', 'Q05', 1, 23),
        (83, 'user1', 'Q16', 0, 10), (84, 'user3', 'Q17', 1, 59),
        (85, 'user1', 'Q13', 1, 19), (86, 'user2', 'Q06', 0, 60),
        (87, 'user5', 'Q10', 1, 11), (88, 'user2', 'Q13', 1, 29),
        (89, 'user3', 'Q10', 0, 22), (90, 'user1', 'Q02', 1, 34),
        (91, 'user3', 'Q03', 0, 45), (92, 'user3', 'Q15', 0, 58),
        (93, 'user2', 'Q15', 1, 28), (94, 'user2', 'Q08', 0, 15),
        (95, 'user4', 'Q05', 0, 28), (96, 'user4', 'Q01', 0, 27),
        (97, 'user4', 'Q10', 1, 18), (98, 'user2', 'Q18', 1, 47),
        (99, 'user5', 'Q11', 0, 22), (100, 'user2', 'Q16', 0, 49),
        (101, 'user3', 'Q09', 1, 48), (102, 'user1', 'Q04', 1, 46),
        (103, 'user1', 'Q07', 0, 9), (104, 'user1', 'Q20', 1, 43),
        (105, 'user3', 'Q11', 1, 45), (106, 'user4', 'Q14', 0, 15),
        (107, 'user2', 'Q07', 1, 39), (108, 'user5', 'Q17', 1, 51),
        (109, 'user3', 'Q06', 1, 20), (110, 'user2', 'Q17', 1, 15),
        (111, 'user5', 'Q04', 1, 34), (112, 'user1', 'Q08', 1, 29),
        (113, 'user3', 'Q07', 1, 22), (114, 'user5', 'Q01', 1, 45),
        (115, 'user3', 'Q13', 1, 49), (116, 'user2', 'Q09', 1, 40),
        (117, 'user4', 'Q12', 1, 19), (118, 'user2', 'Q06', 0, 48),
        (119, 'user4', 'Q03', 1, 25), (120, 'user1', 'Q18', 1, 58),
        (121, 'user1', 'Q05', 1, 54),
    ]

    conn = None
    cursor = None
    try:
        # (수정) .env 파일에서 DB 접속 정보를 안전하게 불러옵니다.
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_DATABASE")

        # (수정) 필수 정보가 있는지 확인
        if not all([db_user, db_password, db_name]):
            print("오류: .env 파일에 DB 접속 정보(DB_USER, DB_PASSWORD, DB_DATABASE)가 없습니다.")
            return

        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        data_to_insert = [
            (user_name, quiz_code, bool(is_correct), time_taken)
            for _, user_name, quiz_code, is_correct, time_taken in history_data
        ]

        sql = """
        INSERT INTO quiz_log (user_id, quiz_id, is_correct, time_taken)
        VALUES (%s, %s, %s, %s)
        """
        if data_to_insert:
            cursor.executemany(sql, data_to_insert)
            conn.commit()
            print(f"✅ 데이터 {cursor.rowcount}건이 성공적으로 삽입되었습니다.")
        else:
            print("⚠️ 삽입할 데이터가 없습니다.")

    except Error as e:
        print(f"❌ 데이터베이스 작업 중 오류가 발생했습니다: {e}")
        if conn:
            conn.rollback()

    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("   - 데이터베이스 연결이 종료되었습니다.")


if __name__ == "__main__":
    insert_quiz_history()
