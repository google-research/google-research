package com.googleresearch.bustle.exception;

/**
 * Represents an issue with the synthesizer, usually indicating a bug.
 */
public final class SynthesisError extends Error {
  public SynthesisError(String message) {
    super(message);
  }
}
