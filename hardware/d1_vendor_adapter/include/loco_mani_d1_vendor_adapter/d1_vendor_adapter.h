#ifndef LOCO_MANI_D1_VENDOR_ADAPTER__D1_VENDOR_ADAPTER_H_
#define LOCO_MANI_D1_VENDOR_ADAPTER__D1_VENDOR_ADAPTER_H_

// This header is deliberately C-compatible.  The implementation is compiled
// against the exact tita_robot/canfd_api.hpp supplied with the D1 image, while
// Python only sees fixed-size arrays and never crosses a C++ ABI boundary.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOCO_MANI_D1_MOTOR_COUNT 16u

typedef struct loco_mani_d1_handle loco_mani_d1_handle_t;

typedef struct loco_mani_d1_motor_state {
  float position;
  float velocity;
  float torque;
  uint16_t status;
} loco_mani_d1_motor_state_t;

typedef struct loco_mani_d1_imu_state {
  uint32_t timestamp;
  float accel[3];
  float gyro[3];
  // The vendor API exposes x/y/z/w.  The adapter intentionally preserves
  // that order; the Python policy boundary converts it to w/x/y/z.
  float quaternion_xyzw[4];
} loco_mani_d1_imu_state_t;

// Construction is the only operation that opens the SocketCAN interface.
// The caller must keep this behind its explicit hardware gate.
loco_mani_d1_handle_t *loco_mani_d1_create(uint32_t motor_count,
                                            const char *can_interface);
void loco_mani_d1_destroy(loco_mani_d1_handle_t *handle);

// Read one coherent vendor snapshot.  All output arrays must contain at least
// LOCO_MANI_D1_MOTOR_COUNT elements.  Return 0 on success, otherwise a
// negative error code; no command is sent by this function.
int loco_mani_d1_read(loco_mani_d1_handle_t *handle,
                      loco_mani_d1_motor_state_t *motors,
                      loco_mani_d1_imu_state_t *imu);

// Send one complete 16-motor packet.  Arrays are in the vendor/API motor
// order, not policy order.  Return 1 when CanfdApi acknowledges the send and
// 0 when it rejects the packet or the handle is invalid.
int loco_mani_d1_send(loco_mani_d1_handle_t *handle,
                      const float *position,
                      const float *velocity,
                      const float *kp,
                      const float *kd,
                      const float *torque);

// Explicitly request the D1 MCU's FORCE_DIRECT mode.  This is a state-changing
// RPC and must never be called as part of ordinary connect/read or shadow mode.
// Return 1 when the vendor API accepts the request, 0 otherwise.
int loco_mani_d1_set_force_direct(loco_mani_d1_handle_t *handle);

int loco_mani_d1_is_motors_timeout(const loco_mani_d1_handle_t *handle);
int loco_mani_d1_is_imu_timeout(const loco_mani_d1_handle_t *handle);

// A thread-local, human-readable diagnostic for the last failed operation.
// The returned pointer remains valid until the next call on the same thread
// or until the handle is destroyed.  It is never used as a control decision.
const char *loco_mani_d1_last_error(const loco_mani_d1_handle_t *handle);

uint32_t loco_mani_d1_feedback_timestamp(const loco_mani_d1_handle_t *handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOCO_MANI_D1_VENDOR_ADAPTER__D1_VENDOR_ADAPTER_H_
