import mysql.connector

def fill_tables():
    # Connect to MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="yourusername",
        password="yourpassword",
        database="yourdatabase"
    )
    
    cursor = conn.cursor()

    # Create the Student table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS Student (
                        Student_id INT PRIMARY KEY,
                        Photo_ID VARCHAR(2045),
                        room_number INT,
                        Contact_number INT NOT NULL,
                        email VARCHAR(40)
                    )''')

    # Example data to insert into the Student table
    student_data = [
        (1, 'photo_001.jpg', 101, 1234567890, 'student1@example.com'),
        (2, 'photo_002.jpg', 102, 2345678901, 'student2@example.com'),
        (3, 'photo_003.jpg', 103, 3456789012, 'student3@example.com'),
        (4, 'photo_004.jpg', 104, 4567890123, 'student4@example.com'),
        (5, 'photo_005.jpg', 105, 5678901234, 'student5@example.com'),
        (6, 'photo_006.jpg', 106, 6789012345, 'student6@example.com'),
        (7, 'photo_007.jpg', 107, 7890123456, 'student7@example.com'),
        (8, 'photo_008.jpg', 108, 8901234567, 'student8@example.com'),
        (9, 'photo_009.jpg', 109, 9012345678, 'student9@example.com'),
        (10, 'photo_010.jpg', 110, 1234567890, 'student10@example.com')
    ]

    # Create the Details table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS Details (
                        entry_id VARCHAR(255) PRIMARY KEY,
                        Student_id INT,
                        date DATE NOT NULL,
                        time TIME NOT NULL,
                        vehicle_number VARCHAR(255),
                        FOREIGN KEY (Student_id) REFERENCES Student(Student_id)
                    )''')

    # Example data to insert into the Details table
    details_data = [
        ('entry_001', 1, '2024-04-01', '08:00:00', 'ABC123'),
        ('entry_002', 2, '2024-04-01', '08:15:00', 'DEF456'),
        ('entry_003', 3, '2024-04-01', '08:30:00', 'GHI789'),
        ('entry_004', 4, '2024-04-01', '08:45:00', 'JKL012'),
        ('entry_005', 5, '2024-04-01', '09:00:00', 'MNO345'),
        ('entry_006', 6, '2024-04-01', '09:15:00', 'PQR678'),
        ('entry_007', 7, '2024-04-01', '09:30:00', 'STU901'),
        ('entry_008', 8, '2024-04-01', '09:45:00', 'VWX234'),
        ('entry_009', 9, '2024-04-01', '10:00:00', 'YZA567'),
        ('entry_010', 10, '2024-04-01', '10:15:00', 'BCD890')
    ]

    # Insert data into the Student table
    cursor.executemany('''INSERT INTO Student (Student_id, Photo_ID, room_number, Contact_number, email)
                          VALUES (%s, %s, %s, %s, %s)''', student_data)

    # Insert data into the Details table
    cursor.executemany('''INSERT INTO Details (entry_id, Student_id, date, time, vehicle_number)
                          VALUES (%s, %s, %s, %s, %s)''', details_data)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Call the fill_tables method to create tables and insert data
fill_tables()
