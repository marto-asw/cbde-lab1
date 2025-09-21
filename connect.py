import psycopg2
from config import load_config

def connect():
    """ Connect to the PostgreSQL database server and return connection """
    try:
        config = load_config()  # obtiene credenciales 
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print("Error connecting to PostgreSQL:", error)
        return None


if __name__ == '__main__':
    conn = connect()
    if conn:
        conn.close()