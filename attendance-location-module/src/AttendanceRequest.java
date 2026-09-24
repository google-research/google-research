package com.example.attendancelocationmodule;

import com.google.gson.annotations.SerializedName;

/**
 * AttendanceRequest - Data model for attendance API request
 * 
 * Contains:
 * - userId: Employee/User ID
 * - latitude: GPS latitude coordinate
 * - longitude: GPS longitude coordinate
 * - timestamp: When the attendance was marked
 * - type: Check-in or Check-out
 */
public class AttendanceRequest {
    @SerializedName("userId")
    private String userId;

    @SerializedName("latitude")
    private double latitude;

    @SerializedName("longitude")
    private double longitude;

    @SerializedName("timestamp")
    private long timestamp;

    @SerializedName("type")
    private String type; // "CHECK_IN" or "CHECK_OUT"

    // Constructor
    public AttendanceRequest(String userId, double latitude, double longitude, String type) {
        this.userId = userId;
        this.latitude = latitude;
        this.longitude = longitude;
        this.timestamp = System.currentTimeMillis();
        this.type = type;
    }

    // Getters and Setters
    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public double getLatitude() {
        return latitude;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    @Override
    public String toString() {
        return "AttendanceRequest{" +
                "userId='" + userId + '\'' +
                ", latitude=" + latitude +
                ", longitude=" + longitude +
                ", timestamp=" + timestamp +
                ", type='" + type + '\'' +
                '}';
    }
}
