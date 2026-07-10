import sqlite3

def check_db():
    conn = sqlite3.connect('cognicore_trajectories.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM trajectories WHERE task_id LIKE '%beam%'")
    print(f"Rows matching beam: {c.fetchone()[0]}")

if __name__ == '__main__':
    check_db()
