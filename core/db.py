# -*- coding: utf-8 -*-
import os

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

def load_data_from_db():
    """
        데이터베이스에서 퀴즈 풀이 기록(quiz_log)을 불러와
        사용자 목록, 퀴즈 목록, 전체 풀이 기록을 반환합니다.
    """

    conn = None
    curcor = None

    try:
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_DATABASE")

        if not all([db_user, db_password, db_name]):
            print("error: .env file is missing")
            return [], [], []

        conn = mysql.connector.connect(
            host=db_host,
            port=3306,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        cursor.execute("SELECT user_id, quiz_id, is_correct, time_taken from quiz_log")
        history = cursor.fetchall()

        if not history:
            print("error: not found quiz log in history")
            return [], [], []

        users = sorted(list(set(row[0] for row in history)))
        quizzes = sorted(list(set(row[1] for row in history)))

        return users, quizzes, history

    except Error as e:
        print(f"error: DB connect or query execution {e}")
        return [], [], []

    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()






