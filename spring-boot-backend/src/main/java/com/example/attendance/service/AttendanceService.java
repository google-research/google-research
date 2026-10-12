package com.example.attendance.service;

import com.example.attendance.dto.AttendanceRequest;
import com.example.attendance.dto.AttendanceResponse;
import com.example.attendance.entity.Attendance;
import com.example.attendance.repository.AttendanceRepository;
import com.example.attendance.util.GeoUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

/**
 * AttendanceService - Business logic layer for attendance operations
 * 
 * Handles:
 * - Check-in/check-out processing
 * - Geofence validation
 * - Attendance record creation/updates
 */
@Service
public class AttendanceService {

    private static final Logger logger = LoggerFactory.getLogger(AttendanceService.class);

    // Geofence Configuration - MALDIVES OFFICE
    private static final double OFFICE_LATITUDE = 8.039313;      // Maldives
    private static final double OFFICE_LONGITUDE = 93.541378;    // Maldives
    private static final double GEOFENCE_RADIUS_METERS = 150;    // 150 meters

    @Autowired
    private AttendanceRepository attendanceRepository;

    /**
     * Process check-in request
     * 1. Validates geofence
     * 2. Creates attendance record
     * 3. Returns response
     */
    public AttendanceResponse processCheckIn(AttendanceRequest request) {
        logger.info("Processing check-in for user: {}", request.getUserId());

        // Check if user is within geofence
        boolean withinGeofence = GeoUtils.isWithinGeofence(
            request.getLatitude(),
            request.getLongitude(),
            OFFICE_LATITUDE,
            OFFICE_LONGITUDE,
            GEOFENCE_RADIUS_METERS
        );

        logger.info("User {} at location ({}, {}) - Within geofence: {}",
            request.getUserId(),
            request.getLatitude(),
            request.getLongitude(),
            withinGeofence);

        // Check if user already has active check-in
        Optional<Attendance> existingRecord = attendanceRepository
            .findByUserIdAndCheckOutTimeIsNull(request.getUserId());

        if (existingRecord.isPresent()) {
            logger.warn("User {} already has active check-in", request.getUserId());
            return new AttendanceResponse(
                false,
                "Already checked in",
                existingRecord.get().getId().toString(),
                withinGeofence,
                "CHECK_IN"
            );
        }

        // Create new attendance record
        Attendance attendance = new Attendance(
            request.getUserId(),
            LocalDateTime.now(),
            request.getLatitude(),
            request.getLongitude(),
            withinGeofence
        );

        Attendance savedAttendance = attendanceRepository.save(attendance);
        logger.info("Check-in successful for user {}: {}", request.getUserId(), savedAttendance.getId());

        String message = withinGeofence ?
            "Checked in successfully" :
            "Checked in from outside allowed location";

        return new AttendanceResponse(
            true,
            message,
            savedAttendance.getId().toString(),
            withinGeofence,
            "CHECK_IN"
        );
    }

    /**
     * Process check-out request
     * 1. Finds active check-in
     * 2. Updates with check-out time
     * 3. Returns response
     */
    public AttendanceResponse processCheckOut(AttendanceRequest request) {
        logger.info("Processing check-out for user: {}", request.getUserId());

        // Check if user has active check-in
        Optional<Attendance> existingRecord = attendanceRepository
            .findByUserIdAndCheckOutTimeIsNull(request.getUserId());

        if (!existingRecord.isPresent()) {
            logger.warn("No active check-in found for user {}", request.getUserId());
            return new AttendanceResponse(
                false,
                "No active check-in found",
                null,
                false,
                "CHECK_OUT"
            );
        }

        // Update check-out time
        Attendance attendance = existingRecord.get();
        attendance.setCheckOutTime(LocalDateTime.now());

        Attendance savedAttendance = attendanceRepository.save(attendance);
        logger.info("Check-out successful for user {}: {}", request.getUserId(), savedAttendance.getId());

        return new AttendanceResponse(
            true,
            "Checked out successfully",
            savedAttendance.getId().toString(),
            attendance.isWithinGeofence(),
            "CHECK_OUT"
        );
    }

    /**
     * Get today's attendance for a user
     */
    public Optional<Attendance> getTodayAttendance(String userId) {
        return attendanceRepository.findByUserIdAndCheckOutTimeIsNull(userId);
    }

    /**
     * Get geofence configuration
     */
    public double[] getGeofenceConfig() {
        return new double[]{
            OFFICE_LATITUDE,
            OFFICE_LONGITUDE,
            GEOFENCE_RADIUS_METERS
        };
    }
}
