package com.example.attendancelocationmodule;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

/**
 * AttendanceAPI - Retrofit interface for backend API calls
 * 
 * Endpoints:
 * - POST /api/attendance/checkin - Submit check-in with location
 * - POST /api/attendance/checkout - Submit check-out with location
 */
public interface AttendanceAPI {

    /**
     * Check-in endpoint
     * Send user location and mark attendance
     */
    @POST("api/attendance/checkin")
    Call<AttendanceResponse> checkIn(@Body AttendanceRequest request);

    /**
     * Check-out endpoint
     * Send user location and mark attendance checkout
     */
    @POST("api/attendance/checkout")
    Call<AttendanceResponse> checkOut(@Body AttendanceRequest request);
}
