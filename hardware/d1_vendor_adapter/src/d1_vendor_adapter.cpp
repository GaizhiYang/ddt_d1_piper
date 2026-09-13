#include "loco_mani_d1_vendor_adapter/d1_vendor_adapter.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tita_robot/canfd_api.hpp"

static_assert(sizeof(can_device::api_motor_in_t) == 16, "unexpected D1 motor input ABI");
static_assert(sizeof(can_device::api_motor_out_t) == 24, "unexpected D1 motor output ABI");
static_assert(sizeof(can_device::api_imu_data_t) == 48, "unexpected D1 IMU ABI");

struct loco_mani_d1_handle {
  explicit loco_mani_d1_handle(uint32_t count, const char *interface)
  : api(count, 0, interface == nullptr ? "can0" : interface), motor_count(count) {}

  can_device::CanfdApi api;
  uint32_t motor_count;
  uint32_t feedback_timestamp{0};
  mutable std::mutex mutex;
  std::string error;
};

namespace {

constexpr int kInvalidArgument = -1;
constexpr int kVendorError = -2;
constexpr int kSizeError = -3;

void set_error(loco_mani_d1_handle_t *handle, const char *message)
{
  if (handle != nullptr) {
    handle->error = message == nullptr ? "unknown D1 adapter error" : message;
  }
}

void set_exception(loco_mani_d1_handle_t *handle, const std::exception &exception)
{
  if (handle != nullptr) {
    handle->error = exception.what();
  }
}

}  // namespace

extern "C" loco_mani_d1_handle_t *loco_mani_d1_create(
  uint32_t motor_count, const char *can_interface)
{
  if (motor_count != LOCO_MANI_D1_MOTOR_COUNT) {
    return nullptr;
  }
  try {
    return new loco_mani_d1_handle(motor_count, can_interface);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void loco_mani_d1_destroy(loco_mani_d1_handle_t *handle)
{
  delete handle;
}

extern "C" int loco_mani_d1_read(
  loco_mani_d1_handle_t *handle,
  loco_mani_d1_motor_state_t *motors,
  loco_mani_d1_imu_state_t *imu)
{
  if (handle == nullptr || motors == nullptr || imu == nullptr) {
    return kInvalidArgument;
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  try {
    const auto *vendor_motors = handle->api.get_motors_in();
    const auto *vendor_status = handle->api.get_motors_status();
    const auto *vendor_imu = handle->api.get_imu_data();
    if (vendor_motors == nullptr || vendor_status == nullptr || vendor_imu == nullptr ||
        vendor_motors->size() != LOCO_MANI_D1_MOTOR_COUNT ||
        vendor_status->size() != LOCO_MANI_D1_MOTOR_COUNT) {
      set_error(handle, "D1 vendor feedback has an unexpected size");
      return kSizeError;
    }
    for (uint32_t index = 0; index < LOCO_MANI_D1_MOTOR_COUNT; ++index) {
      motors[index].position = vendor_motors->at(index).position;
      motors[index].velocity = vendor_motors->at(index).velocity;
      motors[index].torque = vendor_motors->at(index).torque;
      motors[index].status = vendor_status->at(index);
    }
    imu->timestamp = vendor_imu->timestamp;
    handle->feedback_timestamp = vendor_imu->timestamp;
    std::copy(std::begin(vendor_imu->accl), std::end(vendor_imu->accl), imu->accel);
    std::copy(std::begin(vendor_imu->gyro), std::end(vendor_imu->gyro), imu->gyro);
    std::copy(
      std::begin(vendor_imu->quaternion), std::end(vendor_imu->quaternion),
      imu->quaternion_xyzw);
    handle->error.clear();
    return 0;
  } catch (const std::exception &exception) {
    set_exception(handle, exception);
    return kVendorError;
  } catch (...) {
    set_error(handle, "unknown exception while reading D1 feedback");
    return kVendorError;
  }
}

extern "C" int loco_mani_d1_send(
  loco_mani_d1_handle_t *handle,
  const float *position,
  const float *velocity,
  const float *kp,
  const float *kd,
  const float *torque)
{
  if (handle == nullptr || position == nullptr || velocity == nullptr || kp == nullptr ||
      kd == nullptr || torque == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  try {
    std::vector<can_device::api_motor_out_t> motors(LOCO_MANI_D1_MOTOR_COUNT);
    const uint32_t timestamp = can_device::get_current_time();
    for (uint32_t index = 0; index < LOCO_MANI_D1_MOTOR_COUNT; ++index) {
      auto &motor = motors[index];
      motor.timestamp = timestamp;
      motor.position = position[index];
      motor.kp = kp[index];
      motor.velocity = velocity[index];
      motor.kd = kd[index];
      motor.torque = torque[index];
    }
    const bool sent = handle->api.send_motors_can(std::move(motors));
    if (!sent) {
      set_error(handle, "D1 vendor send_motors_can returned false");
      return 0;
    }
    handle->error.clear();
    return 1;
  } catch (const std::exception &exception) {
    set_exception(handle, exception);
    return 0;
  } catch (...) {
    set_error(handle, "unknown exception while sending D1 command");
    return 0;
  }
}

extern "C" int loco_mani_d1_set_force_direct(loco_mani_d1_handle_t *handle)
{
  if (handle == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  try {
    can_device::api_rpc_response_t request{};
    request.timestamp = can_device::get_current_time();
    request.key = can_device::SET_READY_NEXT;
    request.value = can_device::FORCE_DIRECT;
    if (!handle->api.send_command_can_rpc_request(request)) {
      set_error(handle, "D1 vendor FORCE_DIRECT RPC was rejected");
      return 0;
    }
    handle->error.clear();
    return 1;
  } catch (const std::exception &exception) {
    set_exception(handle, exception);
    return 0;
  } catch (...) {
    set_error(handle, "unknown exception while requesting D1 FORCE_DIRECT");
    return 0;
  }
}

extern "C" int loco_mani_d1_is_motors_timeout(const loco_mani_d1_handle_t *handle)
{
  if (handle == nullptr) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  return handle->api.is_motors_timeout() ? 1 : 0;
}

extern "C" int loco_mani_d1_is_imu_timeout(const loco_mani_d1_handle_t *handle)
{
  if (handle == nullptr) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  return handle->api.is_imu_timeout() ? 1 : 0;
}

extern "C" const char *loco_mani_d1_last_error(const loco_mani_d1_handle_t *handle)
{
  static thread_local std::string null_error = "invalid D1 adapter handle";
  static thread_local std::string empty_error;
  if (handle == nullptr) {
    return null_error.c_str();
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  empty_error = handle->error;
  return empty_error.c_str();
}

extern "C" uint32_t loco_mani_d1_feedback_timestamp(
  const loco_mani_d1_handle_t *handle)
{
  if (handle == nullptr) {
    return 0U;
  }
  std::lock_guard<std::mutex> lock(handle->mutex);
  return handle->feedback_timestamp;
}
