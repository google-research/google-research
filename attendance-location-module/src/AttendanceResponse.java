package com.example.attendancelocationmodule;

import com.google.gson.annotations.SerializedName;

/**
 * AttendanceResponse - Data model for API response
 * 
 * Response from server after sending attendance
 */
public class AttendanceResponse {
    @SerializedName("success")
    private boolean success;

    @SerializedName("message")
    private String message;

    @SerializedName("attendanceId")
    private String attendanceId;

    @SerializedName("withinGeofence")
    private boolean withinGeofence;

    // Constructor
    public AttendanceResponse() {
    }

    // Getters and Setters
    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getAttendanceId() {
        return attendanceId;
    }

    public void setAttendanceId(String attendanceId) {
        this.attendanceId = attendanceId;
    }

    public boolean isWithinGeofence() {
        return withinGeofence;
    }

    public void setWithinGeofence(boolean withinGeofence) {
        this.withinGeofence = withinGeofence;
    }

    @Override
    public String toString() {
        return "AttendanceResponse{" +
                "success=" + success +
                ", message='" + message + '\'' +
                ", attendanceId='" + attendanceId + '\'' +
                ", withinGeofence=" + withinGeofence +
                '}';
    }
}
