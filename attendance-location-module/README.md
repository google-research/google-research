# Attendance Location Plugin - Complete Module

This is a ready-to-use Android plugin for location-based attendance tracking. The plugin captures user GPS coordinates and sends them to a Spring Boot backend server for validation and storage.

## 📱 Features

✅ **GPS Location Capture** - Captures current latitude and longitude
✅ **Permission Handling** - Requests and manages location permissions
✅ **API Communication** - Uses Retrofit to send attendance data to backend
✅ **Geofencing Support** - Backend validates if user is within allowed area
✅ **Check-In/Check-Out** - Supports both check-in and check-out attendance
✅ **Error Handling** - Comprehensive error callbacks

## 📦 What's Included

### Android Module Files:
- `build.gradle` - All required dependencies
- `AndroidManifest.xml` - Permissions and manifest
- `LocationManager.java` - GPS location capture
- `AttendanceManager.java` - Main controller class
- `AttendanceService.java` - API communication
- `AttendanceAPI.java` - Retrofit interface
- `AttendanceRequest.java` - Request data model
- `AttendanceResponse.java` - Response data model

### Backend Files:
- `Attendance.java` - JPA Entity
- `AttendanceController.java` - REST endpoints
- `AttendanceService.java` - Business logic
- `GeoUtils.java` - Geofencing calculations

## 🚀 Quick Start

### Step 1: Add to Your Project

Add the module to your app's `settings.gradle`:
```gradle
include ':attendance-location-module'
```

Add dependency to your app's `build.gradle`:
```gradle
dependencies {
    implementation project(':attendance-location-module')
}
```

### Step 2: Request Permissions

In your Activity, add:
```java
private static final int LOCATION_PERMISSION_REQUEST_CODE = 100;

// In onCreate()
String[] permissions = attendanceManager.getRequiredPermissions();
ActivityCompat.requestPermissions(this, permissions, LOCATION_PERMISSION_REQUEST_CODE);

// Handle response
@Override
public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) {
        // Permissions granted, user can now use attendance features
    }
}
```

### Step 3: Initialize AttendanceManager

```java
// Initialize with your backend URL
AttendanceManager manager = new AttendanceManager(
    this, 
    "http://your-backend-server.com:8080/"
);

// Set callback for responses
manager.setCallback(new AttendanceManager.ManagerCallback() {
    @Override
    public void onCheckInSuccess(String message) {
        Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onCheckInFailed(String error) {
        Toast.makeText(MainActivity.this, "Error: " + error, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onCheckOutSuccess(String message) {
        Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onCheckOutFailed(String error) {
        Toast.makeText(MainActivity.this, "Error: " + error, Toast.LENGTH_SHORT).show();
    }
});
```

### Step 4: Mark Attendance

```java
// Check-in
manager.markCheckIn("USER_ID_12345");

// Check-out
manager.markCheckOut("USER_ID_12345");
```

## 🔧 Backend Configuration

See `spring-boot-backend/` directory for complete Spring Boot implementation.

## 📍 Geofencing

The backend validates that the user is within a certain distance (default: 150 meters) of the office location.

**Default Office Location:**
- Latitude: 40.7128 (New York)
- Longitude: -74.0060
- Radius: 150 meters

Modify in `AttendanceController.java`:
```java
private static final double OFFICE_LAT = 40.7128;
private static final double OFFICE_LON = -74.0060;
private static final double GEOFENCE_RADIUS = 150;
```

## 📚 API Endpoints

### POST `/api/attendance/checkin`
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

**Response:**
```json
{
    "success": true,
    "message": "Checked in successfully",
    "attendanceId": "ATT_123456",
    "withinGeofence": true,
    "type": "CHECK_IN"
}
```

### POST `/api/attendance/checkout`
Same as check-in but with `type: "CHECK_OUT"`

## 🔐 Permissions Required

Add to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.INTERNET" />
```

## 📝 Dependencies

- Google Play Services (Location): v21.0.1
- Retrofit: v2.9.0
- Gson: v2.10.1
- AndroidX Core: v1.10.1

## 🆘 Troubleshooting

### Location returning null
- Ensure device has GPS enabled
- Ensure location permissions are granted
- Try using `getLastKnownLocation()` first

### Backend not responding
- Check if server URL is correct and accessible
- Ensure backend is running on specified port
- Check network connectivity

### "Not within allowed location" error
- User is outside the geofence radius
- Move closer to the office location
- Check geofence coordinates in backend

## 📞 Support

For issues or questions, check the example implementation in `USAGE_EXAMPLE.md`

---

**Version:** 1.0.0  
**Last Updated:** 2026-05-25
