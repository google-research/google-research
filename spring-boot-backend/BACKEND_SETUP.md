# Attendance Location Plugin - Backend Configuration

## 🔧 Spring Boot Backend Setup

Complete Spring Boot application for location-based attendance tracking backend.

## 📋 Files Included

### Entities
- `Attendance.java` - JPA entity for database

### DTOs
- `AttendanceRequest.java` - Request model from Android
- `AttendanceResponse.java` - Response model to Android

### Business Logic
- `AttendanceService.java` - Core business logic
- `GeoUtils.java` - Geofencing calculations (Haversine formula)

### Data Access
- `AttendanceRepository.java` - JPA repository interface

### API
- `AttendanceController.java` - REST endpoints

## 🚀 Quick Start

### Step 1: Create Spring Boot Project

Use Spring Initializr (https://start.spring.io/):
- **Project:** Maven
- **Language:** Java
- **Spring Boot:** 2.7.x or 3.x
- **Dependencies:**
  - Spring Web
  - Spring Data JPA
  - MySQL Driver (or PostgreSQL)
  - Lombok (optional)

### Step 2: Add Dependencies

In `pom.xml`:
```xml
<dependencies>
    <!-- Spring Boot Starters -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    
    <!-- Database -->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <version>8.0.33</version>
    </dependency>
    
    <!-- Testing -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
```

### Step 3: Configure Database

In `application.properties`:
```properties
# Server
server.port=8080

# MySQL Database
spring.datasource.url=jdbc:mysql://localhost:3306/attendance_db?useSSL=false&serverTimezone=UTC
spring.datasource.username=root
spring.datasource.password=your_password
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver

# JPA/Hibernate
spring.jpa.database-platform=org.hibernate.dialect.MySQL8Dialect
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true

# Logging
logging.level.root=INFO
logging.level.com.example.attendance=DEBUG
```

### Step 4: Create Application Class

```java
package com.example.attendance;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class AttendanceApplication {
    public static void main(String[] args) {
        SpringApplication.run(AttendanceApplication.class, args);
    }
}
```

### Step 5: Copy Backend Files

Copy all files from `spring-boot-backend/src/` to your project's `src/` directory

## 📚 API Endpoints

### Check-In
**POST** `/api/attendance/checkin`

**Request:**
```json
{
    "userId": "EMP001",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": 1622505600000,
    "type": "CHECK_IN"
}
```

**Response (Success):**
```json
{
    "success": true,
    "message": "Checked in successfully",
    "attendanceId": "123",
    "withinGeofence": true,
    "type": "CHECK_IN"
}
```

**Response (Outside Geofence):**
```json
{
    "success": true,
    "message": "Checked in from outside allowed location",
    "attendanceId": "123",
    "withinGeofence": false,
    "type": "CHECK_IN"
}
```

### Check-Out
**POST** `/api/attendance/checkout`

Same request format as check-in with `type: "CHECK_OUT"`

### Health Check
**GET** `/api/attendance/health`

**Response:**
```
Attendance API is running
```

## 📍 Geofence Configuration

Default office location (in `AttendanceService.java`):
```java
private static final double OFFICE_LATITUDE = 40.7128;  // New York
private static final double OFFICE_LONGITUDE = -74.0060;
private static final double GEOFENCE_RADIUS_METERS = 150;
```

**To change geofence:**
1. Update values in `AttendanceService.java`
2. Recompile and restart server

## 🗄️ Database Schema

### Attendance Table
```sql
CREATE TABLE attendance (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(255) NOT NULL,
    check_in_time DATETIME NOT NULL,
    check_out_time DATETIME,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    within_geofence BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    
    INDEX idx_user_id (user_id),
    INDEX idx_check_in_time (check_in_time)
);
```

## 🏃 Running the Application

```bash
# Using Maven
mvn spring-boot:run

# Using JAR
java -jar attendance-api.jar

# Application will start at http://localhost:8080
```

## ✅ Testing the API

### Using cURL

**Check-In:**
```bash
curl -X POST http://localhost:8080/api/attendance/checkin \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "EMP001",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": 1622505600000,
    "type": "CHECK_IN"
  }'
```

**Check-Out:**
```bash
curl -X POST http://localhost:8080/api/attendance/checkout \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "EMP001",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": 1622505600000,
    "type": "CHECK_OUT"
  }'
```

**Health Check:**
```bash
curl http://localhost:8080/api/attendance/health
```

### Using Postman
1. Create POST request to `http://localhost:8080/api/attendance/checkin`
2. Set Body as JSON
3. Paste request JSON
4. Click Send

## 📊 Database Queries

### View all attendance records
```sql
SELECT * FROM attendance ORDER BY created_at DESC;
```

### View employee's attendance history
```sql
SELECT * FROM attendance 
WHERE user_id = 'EMP001' 
ORDER BY check_in_time DESC;
```

### View today's check-ins outside geofence
```sql
SELECT * FROM attendance 
WHERE DATE(check_in_time) = CURDATE() 
AND within_geofence = false;
```

## 🔐 Security Notes

- Add authentication (JWT, OAuth2) for production
- Validate user permissions before accessing attendance data
- Use HTTPS for production deployment
- Store sensitive data securely

## 🆘 Troubleshooting

### Database connection error
- Ensure MySQL is running
- Check username/password in `application.properties`
- Verify database name

### Port already in use
- Change `server.port` in `application.properties`
- Or kill process using port 8080

### Android can't connect to backend
- Check if server URL is correct in Android app
- Ensure server is accessible from device/emulator
- For emulator: Use `10.0.2.2` instead of `localhost`

---

**Version:** 1.0.0  
**Last Updated:** 2026-05-25
