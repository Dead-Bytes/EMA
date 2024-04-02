import mysql.connector
import Entries
def check_database_exists():
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host="your_host",
            user="your_username",
            password="your_password"
        )

        # Create a cursor object
        cursor = connection.cursor()

        # Check if the database exists
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        for database in databases:
            if database[0] == 'EMA':
                return True  # Database exists
        return False  # Database doesn't exist

    except mysql.connector.Error as error:
        print("Error:", error)
        return False

def instantiate_database():
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host="your_host",
            user="your_username",
            password="your_password"
        )

        # Create a cursor object
        cursor = connection.cursor()

        # Create EMA database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS EMA")

        # Use the EMA database
        cursor.execute("USE EMA")

        # Create student table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Student (
                Student_id INT PRIMARY KEY,
                Photo_ID VARCHAR(2045),
                room_number INT,
                Contact_number INT NOT NULL,
                email VARCHAR(40)
            )
        """)

        # Create details table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Details (
                entry_id VARCHAR(255) PRIMARY KEY,
                Student_id INT,
                date DATE NOT NULL,
                time TIME NOT NULL,
                vehicle_number VARCHAR(255),
                FOREIGN KEY (Student_id) REFERENCES Student(Student_id)
            )
        """)

        # Call the function to fill the database
        fill_database()

        return True

    except mysql.connector.Error as error:
        print("Error:", error)
        return False

def fill_database():
    # Call the function from your Database python file to fill the database
    # You need to replace the following line with your actual function call
    Entries.fill_database()

if __name__ == "__main__":
    if check_database_exists():
        print("EMA database already exists.")
    else:
        print("EMA database does not exist. Instantiating...")
        if instantiate_database():
            print("EMA database instantiated successfully.")
        else:
            print("Failed to instantiate EMA database.")
