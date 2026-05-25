# 📱 Complete Integration Guide - Attendance Location Plugin

This guide walks you through integrating and deploying the complete attendance tracking system.

## 🎯 What You Have

✅ **Android Plugin Module** - Location capture + API client
✅ **Spring Boot Backend** - REST API + Geofencing  
✅ **Complete Documentation** - Setup & usage guides

---

## 📦 Part 1: Android Setup

### Step 1.1: Add Module to Your Project

1. Copy `attendance-location-module/` folder into your Android project root
2. Edit `settings.gradle` in project root:
```gradle
include ':app'
include ':attendance-location-module'
```

3. In `app/build.gradle`, add dependency:
```gradle
dependencies {
    implementation project(':attendance-location-module')
    // ... other deps
}
```

### Step 1.2: Request Permissions

In your `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.INTERNET" />
```

### Step 1.3: Create UI Layout

Create `activity_attendance.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Mark Your Attendance"
        android:textSize="24sp"
        android:textStyle="bold"
        android:gravity="center"
        android:layout_marginBottom="32dp" />

    <TextView
        android:id="@+id/tv_status"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Status: Ready"
        android:textSize="16sp"
        android:padding="16dp"
        android:background="#f0f0f0"
        android:layout_marginBottom="32dp" />

    <Button
        android:id="@+id/btn_check_in"
        android:layout_width="match_parent"
        android:layout_height="60dp"
        android:text="CHECK IN"
        android:textSize="18sp"
        android:layout_marginBottom="16dp" />

    <Button
        android:id="@+id/btn_check_out"
        android:layout_width="match_parent"
        android:layout_height="60dp"
        android:text="CHECK OUT"
        android:textSize="18sp" />

</LinearLayout>
```

### Step 1.4: Implement Activity

Create `AttendanceActivity.java`:

```java
package com.example.myapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.example.attendancelocationmodule.AttendanceManager;

public class AttendanceActivity extends AppCompatActivity {
    
    private static final int LOCATION_PERMISSION_REQUEST_CODE = 100;
    
    private AttendanceManager attendanceManager;
    private Button btnCheckIn, btnCheckOut;
    private TextView tvStatus;
    private String userId = "EMP12345";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_attendance);

        btnCheckIn = findViewById(R.id.btn_check_in);
        btnCheckOut = findViewById(R.id.btn_check_out);
        tvStatus = findViewById(R.id.tv_status);

        // Initialize with YOUR backend URL
        attendanceManager = new AttendanceManager(
            this,
            "http://192.168.1.100:8080/"  // Change this!
        );

        attendanceManager.setCallback(new AttendanceManager.ManagerCallback() {
            @Override
            public void onCheckInSuccess(String message) {
                tvStatus.setText("✅ " + message);
                btnCheckIn.setEnabled(false);
                btnCheckOut.setEnabled(true);
            }

            @Override
            public void onCheckInFailed(String error) {
                tvStatus.setText("❌ " + error);
            }

            @Override
            public void onCheckOutSuccess(String message) {
                tvStatus.setText("✅ " + message);
                btnCheckIn.setEnabled(true);
                btnCheckOut.setEnabled(false);
            }

            @Override
            public void onCheckOutFailed(String error) {
                tvStatus.setText("❌ " + error);
            }
        });

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestLocationPermissions();
        }

        btnCheckIn.setOnClickListener(v -> {
            if (attendanceManager.hasLocationPermission()) {
                tvStatus.setText("Capturing location...");
                attendanceManager.markCheckIn(userId);
            } else {
                requestLocationPermissions();
            }
        });

        btnCheckOut.setOnClickListener(v -> {
            if (attendanceManager.hasLocationPermission()) {
                tvStatus.setText("Capturing location...");
                attendanceManager.markCheckOut(userId);
            } else {
                requestLocationPermissions();
            }
        });

        btnCheckOut.setEnabled(false);
    }

    private void requestLocationPermissions() {
        String[] permissions = attendanceManager.getRequiredPermissions();
        ActivityCompat.requestPermissions(this, permissions, LOCATION_PERMISSION_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                tvStatus.setText("✅ Location permissions granted");
            } else {
                tvStatus.setText("❌ Location permissions denied");
            }
        }
    }
}
```

---

## 🔧 Part 2: Spring Boot Backend Setup

### Step 2.1: Create Project

Go to **https://start.spring.io**:
- **Project:** Maven
- **Language:** Java
- **Spring Boot:** 2.7.x or 3.1.x
- **Dependencies:**
  - Spring Web
  - Spring Data JPA
  - MySQL Driver
  
Click **GENERATE** and extract the ZIP

### Step 2.2: Update `pom.xml`

Add these dependencies inside `<dependencies>`:
```xml
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <version>8.0.33</version>
</dependency>
```

### Step 2.3: Configure Database

Create `application.properties`:
```properties
server.port=8080

spring.datasource.url=jdbc:mysql://localhost:3306/attendance_db?useSSL=false&serverTimezone=UTC
spring.datasource.username=root
spring.datasource.password=your_password
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver

spring.jpa.database-platform=org.hibernate.dialect.MySQL8Dialect
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=false

logging.level.com.example.attendance=DEBUG
```

### Step 2.4: Copy Backend Files

Copy all folders from `spring-boot-backend/src/main/java/com/example/attendance/` to your project's `src/main/java/com/example/attendance/`

### Step 2.5: Create Database

In MySQL:
```sql
CREATE DATABASE attendance_db;
USE attendance_db;
```

### Step 2.6: Run Backend

```bash
mvn spring-boot:run
```

You should see:
```
Started AttendanceApplication in 5.234 seconds
```

Test with:
```bash
curl http://localhost:8080/api/attendance/health
# Response: Attendance API is running
```

---

## 🔗 Part 3: Connect Android to Backend

### Key Points:

1. **Update Backend URL in Android:**
   ```java
   attendanceManager = new AttendanceManager(
       this,
       "http://192.168.1.100:8080/"  // Use your server IP
   );
   ```

2. **For Android Emulator:**
   - Use `http://10.0.2.2:8080/` instead of `localhost`

3. **For Physical Device:**
   - Ensure device and server are on same network
   - Use server's local IP (e.g., `192.168.x.x`)

4. **For Production:**
   - Use HTTPS
   - Use domain name (e.g., `https://api.mycompany.com/`)

---

## 🧪 Part 4: Testing

### Test Scenario 1: Normal Check-In

1. **Start Backend:**
   ```bash
   mvn spring-boot:run
   ```

2. **Run Android App** on emulator or device

3. **Click "CHECK IN"**
   - App captures GPS location
   - Sends to backend
   - Backend validates geofence
   - Returns success response

4. **Check Database:**
   ```sql
   SELECT * FROM attendance ORDER BY created_at DESC LIMIT 1;
   ```

### Test Scenario 2: Outside Geofence

1. On Android emulator, set location outside office (in Extended Controls → Location)
2. Click "CHECK IN"
3. You should see: "Checked in from outside allowed location"
4. Check database: `within_geofence` should be `false`

### Test Scenario 3: Check-Out

1. After check-in, click "CHECK OUT"
2. Check database: `check_out_time` should be populated

---

## 📊 Monitoring

### View All Attendance Records
```sql
SELECT 
    id,
    user_id,
    DATE(check_in_time) as date,
    TIME(check_in_time) as check_in_time,
    TIME(check_out_time) as check_out_time,
    within_geofence
FROM attendance
ORDER BY check_in_time DESC;
```

### View Employee Report
```sql
SELECT 
    user_id,
    COUNT(*) as total_days,
    SUM(CASE WHEN within_geofence = true THEN 1 ELSE 0 END) as on_site_days,
    SUM(CASE WHEN within_geofence = false THEN 1 ELSE 0 END) as off_site_days
FROM attendance
GROUP BY user_id;
```

---

## 🚀 Deployment

### Android App
1. Build APK: `Build → Build Bundle(s) / APK(s) → Build APK(s)`
2. Test on real device
3. Publish to Google Play Store

### Spring Boot Backend

**Option 1: Local Server**
```bash
mvn clean package
java -jar target/attendance-0.0.1-SNAPSHOT.jar
```

**Option 2: Docker**
```dockerfile
FROM openjdk:11-jre-slim
COPY target/attendance-0.0.1-SNAPSHOT.jar app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
```

**Option 3: Cloud (AWS/Azure/GCP)**
- Deploy JAR to cloud server
- Configure database in cloud
- Update Android app with cloud URL

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| "Not within allowed location" | Move closer to office or update geofence in `AttendanceService.java` |
| Location returns null | Enable GPS on device, wait for signal |
| Backend not responding | Check if server running, verify URL in app |
| Database connection error | Check MySQL is running, verify credentials |
| Port 8080 already in use | Change `server.port` in `application.properties` |

---

## 📞 Next Steps

1. ✅ **Customize geofence location** (latitude/longitude in backend)
2. ✅ **Add authentication** (JWT tokens for security)
3. ✅ **Add admin dashboard** (view attendance reports)
4. ✅ **Mobile notifications** (alert on check-in/out)
5. ✅ **Analytics** (attendance patterns, late arrivals)

---

**Total Setup Time:** ~30-45 minutes  
**Version:** 1.0.0  
**Support:** Check README files in each directory

