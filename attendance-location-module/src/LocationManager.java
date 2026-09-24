package com.example.attendancelocationmodule;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Location;
import android.util.Log;

import androidx.core.app.ActivityCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;
import com.google.android.gms.tasks.OnSuccessListener;

/**
 * LocationManager - Handles GPS location capture
 * 
 * This class provides methods to:
 * - Request location permissions
 * - Capture current user location (latitude and longitude)
 * - Handle location callbacks
 */
public class LocationManager {
    private static final String TAG = "LocationManager";
    private FusedLocationProviderClient fusedLocationClient;
    private Context context;
    private LocationCallback locationCallback;

    public interface LocationCallback {
        void onLocationReceived(double latitude, double longitude);
        void onLocationError(String error);
    }

    public LocationManager(Context context) {
        this.context = context;
        this.fusedLocationClient = LocationServices.getFusedLocationProviderClient(context);
    }

    /**
     * Set callback to receive location updates
     */
    public void setLocationCallback(LocationCallback callback) {
        this.locationCallback = callback;
    }

    /**
     * Check if location permissions are granted
     */
    public boolean hasLocationPermission() {
        return ActivityCompat.checkSelfPermission(context, 
            Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
            && ActivityCompat.checkSelfPermission(context, 
            Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Get required permissions array for requesting
     */
    public String[] getRequiredPermissions() {
        return new String[]{
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        };
    }

    /**
     * Capture current location asynchronously
     */
    public void getCurrentLocation() {
        if (!hasLocationPermission()) {
            if (locationCallback != null) {
                locationCallback.onLocationError("Location permissions not granted");
            }
            Log.w(TAG, "Location permissions not granted");
            return;
        }

        fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
            .addOnSuccessListener(location -> {
                if (location != null) {
                    double latitude = location.getLatitude();
                    double longitude = location.getLongitude();
                    
                    Log.d(TAG, "Location captured - Lat: " + latitude + ", Lon: " + longitude);
                    
                    if (locationCallback != null) {
                        locationCallback.onLocationReceived(latitude, longitude);
                    }
                } else {
                    if (locationCallback != null) {
                        locationCallback.onLocationError("Unable to get current location");
                    }
                    Log.w(TAG, "Unable to get current location");
                }
            })
            .addOnFailureListener(e -> {
                Log.e(TAG, "Error getting location", e);
                if (locationCallback != null) {
                    locationCallback.onLocationError(e.getMessage());
                }
            });
    }

    /**
     * Get last known location (faster but may be outdated)
     */
    public void getLastKnownLocation() {
        if (!hasLocationPermission()) {
            if (locationCallback != null) {
                locationCallback.onLocationError("Location permissions not granted");
            }
            return;
        }

        fusedLocationClient.getLastLocation()
            .addOnSuccessListener(location -> {
                if (location != null) {
                    double latitude = location.getLatitude();
                    double longitude = location.getLongitude();
                    
                    Log.d(TAG, "Last known location - Lat: " + latitude + ", Lon: " + longitude);
                    
                    if (locationCallback != null) {
                        locationCallback.onLocationReceived(latitude, longitude);
                    }
                } else {
                    if (locationCallback != null) {
                        locationCallback.onLocationError("No last known location available");
                    }
                }
            })
            .addOnFailureListener(e -> {
                Log.e(TAG, "Error getting last known location", e);
                if (locationCallback != null) {
                    locationCallback.onLocationError(e.getMessage());
                }
            });
    }
}
