// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cerrno>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <controller_manager/controller_manager.hpp>
#include <pluginlib/class_loader.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <mujoco/mujoco.h>
#include "glfw_adapter.h"
#include "simulate.h"
#include "array_safety.h"
#include "mujoco_sim_ros2/mujoco_physics_plugin.hpp"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
  #include <windows.h>
#else
  #if defined(__APPLE__)
    #include <mach-o/dyld.h>
  #endif
  #include <sys/errno.h>
  #include <unistd.h>
#endif
}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// MuJoCo's stock ``simulate`` application assumes a GLFW window.  ROS 2
// launch must also be usable on Jetson/CI machines without a display, so the
// physics/plugin path gets a small no-op UI adapter in headless mode.  The
// adapter is never asked to render; it only supplies the state object owned by
// ``Simulate`` and satisfies its platform abstraction.
class HeadlessUIAdapter final : public mj::PlatformUIAdapter {
 public:
  std::pair<double, double> GetCursorPosition() const override { return {0.0, 0.0}; }
  double GetDisplayPixelsPerInch() const override { return 96.0; }
  std::pair<int, int> GetFramebufferSize() const override { return {1, 1}; }
  std::pair<int, int> GetWindowSize() const override { return {1, 1}; }
  bool IsGPUAccelerated() const override { return false; }
  void PollEvents() override {}
  void SetClipboardString(const char*) override {}
  void SetVSync(bool) override {}
  void SetWindowTitle(const char*) override {}
  bool ShouldCloseWindow() const override { return false; }
  void SwapBuffers() override {}
  void ToggleFullscreen() override {}
  bool IsLeftMouseButtonPressed() const override { return false; }
  bool IsMiddleMouseButtonPressed() const override { return false; }
  bool IsRightMouseButtonPressed() const override { return false; }
  bool IsAltKeyPressed() const override { return false; }
  bool IsCtrlKeyPressed() const override { return false; }
  bool IsShiftKeyPressed() const override { return false; }
  bool IsMouseButtonDownEvent(int) const override { return false; }
  bool IsKeyDownEvent(int) const override { return false; }
  int TranslateKeyCode(int key) const override { return key; }
  mjtButton TranslateMouseButton(int) const override { return mjBUTTON_NONE; }
};

// constants
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
const int kErrorLength = 1024;          // load error string length

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

// The renderer and physics loop own the MuJoCo model/data lifetime.  Keep a
// pointer to the simulation UI only so SIGINT can request an orderly shutdown;
// deleting ``m``/``d`` directly from the signal handler races with the physics
// thread and used to cause a double free (exit code -11 on normal Ctrl-C).
std::atomic<mj::Simulate*> active_sim{nullptr};

using Seconds = std::chrono::duration<double>;

std::unique_ptr<pluginlib::ClassLoader<mujoco_sim_ros2::MujocoPhysicsPlugin>> physics_plugin_loader;
std::vector<std::shared_ptr<mujoco_sim_ros2::MujocoPhysicsPlugin>> physics_plugins;

//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new(std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *=2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new(std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}



// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif


  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}


//------------------------------------------- simulation -------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

// Keep the ros2_control hook ordering identical for every physics step.  In
// particular, the first step after a real-time re-sync must not bypass
// ControllerManager::update(); doing so creates a one-step stale command and
// makes sim2sim traces differ from the regular loop.
void StepWithPlugins(
    mjModel* model, mjData* data,
    std::vector<std::shared_ptr<mujoco_sim_ros2::MujocoPhysicsPlugin>>& plugins) {
  if (plugins.empty()) {
    mj_step(model, data);
    return;
  }
  for (auto& plugin : plugins) {
    plugin->PreUpdate(model, data);
  }
  mj_step1(model, data);
  for (auto& plugin : plugins) {
    plugin->Update(model, data);
  }
  mj_step2(model, data);
  for (auto& plugin : plugins) {
    plugin->PostUpdate(model, data);
  }
}

// Service controller-manager callbacks while the physics model is held at
// its initial state.  ros2_control performs controller activation in its
// update loop, so merely setting ``sim.run = 0`` would make every spawner time
// out.  No MuJoCo integration step is performed here; only the plugin hooks
// (and therefore the embedded ControllerManager) are serviced.
void UpdatePluginsWithoutPhysics(
    mjModel* model, mjData* data,
    std::vector<std::shared_ptr<mujoco_sim_ros2::MujocoPhysicsPlugin>>& plugins) {
  for (auto& plugin : plugins) {
    plugin->PreUpdate(model, data);
  }
  for (auto& plugin : plugins) {
    plugin->Update(model, data);
  }
  for (auto& plugin : plugins) {
    plugin->PostUpdate(model, data);
  }
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj::Simulate::Clock::now();
  if (mju::strlen_arr(filename)>4 &&
      !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                    mju::sizeof_arr(filename) - mju::strlen_arr(filename)+4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length-1] == '\n') {
        loadError[error_length-1] = '\0';
      }
    }
  }
  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  else if (load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);

  return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim,
  std::vector<std::shared_ptr<mujoco_sim_ros2::MujocoPhysicsPlugin>>& plugins,
  bool real_time, double duration, const rclcpp::Node::SharedPtr& node,
  bool start_paused) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    // In a launch-composed ROS graph, keep the model paused until the
    // controller spawners and policy node are ready.  The launch file flips
    // this parameter through the normal ROS parameter service.  This avoids
    // consuming simulation time (or letting an uncontrolled robot fall) while
    // controller_manager is still discovering services.
    if (start_paused) {
      bool paused = true;
      try {
        paused = node->get_parameter("start_paused").as_bool();
      } catch (const std::exception &) {
        // The parameter is declared by main before this thread starts.  Keep
        // the safe default if a shutdown races parameter access.
        paused = true;
      }
      if (paused) {
        std::unique_lock<std::recursive_mutex> lock(sim.mtx);
        if (m && d) {
          UpdatePluginsWithoutPhysics(m, d, plugins);
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
    }
    // ``duration`` is expressed in MuJoCo simulation seconds, rather than
    // wall-clock seconds.  This keeps fast/offline runs and real-time runs
    // semantically identical.  A value of zero disables the limit.
    if (duration > 0.0 && d && d->time >= duration) {
      sim.exitrequest.store(1);
      break;
    }
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    // In fast/headless mode do not use the GUI synchronisation clock.  This
    // is the same single-step/plugin ordering as the real-time path, but it
    // advances as quickly as the host permits and is useful for CI/trace
    // generation.  The controller-manager executor remains on its own
    // thread, so ROS services continue to be serviced.
    if (!real_time) {
      std::unique_lock<std::recursive_mutex> lock(sim.mtx);
      if (m && sim.run) {
        StepWithPlugins(m, d, plugins);
        const char* message = Diverged(m->opt.disableflags, d);
        if (message) {
          sim.run = 0;
          mju::strcpy_arr(sim.load_error, message);
        } else {
          sim.AddToHistory();
        }
      }
      lock.unlock();
      std::this_thread::yield();
      continue;
    }

    // sleep for 1 ms or yield, to let the render thread run
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncmisalign
          bool misaligned =
              std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            StepWithPlugins(m, d, plugins);
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction/sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim)*slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise();

              // call mj_step with the same plugin ordering as the first step
              StepWithPlugins(m, d, plugins);

              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}
}  // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate* sim, rclcpp::Node::SharedPtr node,
                   rclcpp::NodeOptions node_options,
                   const char* filename,
                   const std::vector<std::string>& physics_plugin_names,
                   bool headless, bool real_time, double duration,
                   bool start_paused) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    if (!headless) {
      sim->LoadMessage(filename);
    }
    m = LoadModel(filename, *sim);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
    }
    if (d) {
      if (headless) {
        // There is no render thread to consume Simulate::Load() in headless
        // mode.  Publish the model/data pointers directly instead.
        const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
        sim->m_ = m;
        sim->d_ = d;
        sim->loadrequest = 0;
      } else {
        sim->Load(m, d, filename);
      }

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m, d);

    } else {
      sim->LoadMessageClear();
    }
  }

  if (!physics_plugin_names.empty()) {
    bool success = true;
    physics_plugin_loader = std::make_unique<pluginlib::ClassLoader<mujoco_sim_ros2::MujocoPhysicsPlugin>>
      ("mujoco_sim_ros2", "mujoco_sim_ros2::MujocoPhysicsPlugin");
    try {
      for (const auto& plugin_name : physics_plugin_names) {
        physics_plugins.push_back(physics_plugin_loader->createSharedInstance(plugin_name));
      }
    } catch(pluginlib::PluginlibException& ex) {
      printf("The plugin failed to load for some reason. \nError: %s\n", ex.what());
      success = false;
    }
    if (success) {
      std::cout << "Successfully loaded " << physics_plugins.size()
               << " physics plugin(s)" << std::endl;
    }
  }

  for (auto& plugin : physics_plugins) {
    plugin->Configure(node, node_options, m, d);
  }

  PhysicsLoop(*sim, physics_plugins, real_time, duration, node, start_paused);

  // Model/data are released by the owning main thread after the embedded
  // ros2_control plugin (and its ControllerManager/hardware objects) has
  // been destroyed.  Releasing them here leaves dangling pointers in
  // MujocoSystem during controller shutdown and can crash in DDS/plugin
  // teardown.
}

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char** argv) {

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER!=mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // install signal handler
  //--------------------- set up ros node ---------------------//
  // Let the process handle SIGINT itself.  rclcpp's default handler shuts the
  // global context down immediately, while the MuJoCo physics thread still
  // owns controller-manager nodes; disabling it lets the main thread request
  // the UI exit, join the physics thread, and only then shut ROS down.
  rclcpp::InitOptions init_options;
  init_options.shutdown_on_signal = false;
  rclcpp::init(argc, argv, init_options, rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, [](int) {
    if (auto* sim = active_sim.load()) {
      sim->exitrequest.store(1);
    }
  });
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared(
      "mujoco_sim_ros2_node");

  // get the ros arg, mainly for getting --param-file for cm
  rclcpp::NodeOptions cm_node_options = controller_manager::get_cm_node_options();
  std::vector<std::string> node_arguments = cm_node_options.arguments();
  for(int i = 1; i < argc; ++i)
  {
    if(node_arguments.empty() && std::string(argv[i]) != "--ros-args") continue;
    node_arguments.emplace_back(argv[i]);
  }
  cm_node_options.arguments(node_arguments);

  // declare parameters
  node->declare_parameter("model_package", "");
  node->declare_parameter("model_file", "");
  node->declare_parameter("physics_plugins", std::vector<std::string>());
  node->declare_parameter("headless", false);
  node->declare_parameter("real_time", true);
  node->declare_parameter("duration", 0.0);
  node->declare_parameter("start_paused", false);

  // get parameters
  std::string model_pkg =
      node->get_parameter("model_package").get_parameter_value().get<std::string>();
  std::string model_file =
      node->get_parameter("model_file").get_parameter_value().get<std::string>();
  std::vector<std::string> physics_plugin_names =
      node->get_parameter("physics_plugins").get_parameter_value().get<std::vector<std::string>>();
  const bool headless = node->get_parameter("headless").as_bool();
  const bool real_time = node->get_parameter("real_time").as_bool();
  const double duration = node->get_parameter("duration").as_double();
  const bool start_paused = node->get_parameter("start_paused").as_bool();
  if (!std::isfinite(duration) || duration < 0.0) {
    std::cerr << "duration must be finite and non-negative (0 means unlimited)" << std::endl;
    return -1;
  }

  std::string package_share_path;
  try {
    package_share_path =
        ament_index_cpp::get_package_share_directory(model_pkg);
    // std::cout << "Shared folder path of package '" << model_pkg
    //           << "': " << package_share_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: Unable to find package '" << model_pkg
              << "'. Ensure it is installed and sourced." << std::endl;
    return -1;
  }
  model_file = package_share_path + "/" + model_file;

  std::cout << "========================================" << std::endl;
  std::cout << "Loaded ROS parameters:" << std::endl;
  std::cout << "model package: " << model_pkg << std::endl;
  std::cout << "model file: " << model_file << std::endl;
  std::cout << "physics plugins: " << std::endl;
  if (physics_plugin_names.empty()) {
    std::cout << "  - none" << std::endl;
  }
  for (const auto &plugin : physics_plugin_names) {
    std::cout << "  - " << plugin << std::endl;
  }
  std::cout << "headless: " << (headless ? "true" : "false") << std::endl;
  std::cout << "real_time: " << (real_time ? "true" : "false") << std::endl;
  std::cout << "duration: " << duration << " s (simulation time)" << std::endl;
  std::cout << "start_paused: " << (start_paused ? "true" : "false") << std::endl;
  std::cout << "========================================" << std::endl;

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // simulate object encapsulates the UI
  std::unique_ptr<mj::PlatformUIAdapter> ui_adapter;
  if (headless) {
    ui_adapter = std::make_unique<HeadlessUIAdapter>();
  } else {
    ui_adapter = std::make_unique<mj::GlfwAdapter>();
  }
  auto sim = std::make_unique<mj::Simulate>(
      std::move(ui_adapter),
      &cam, &opt, &pert, /* is_passive = */ false
  );
  active_sim.store(sim.get());

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(),
  node, cm_node_options, model_file.c_str(), physics_plugin_names,
  headless, real_time, duration, start_paused);

  // The normal mode renders on the main thread.  In headless mode there is no
  // GLFW event loop; wait for SIGINT/ROS shutdown while the physics thread
  // advances the model.  This keeps the same process/node ownership and makes
  // ``headless:=true`` usable over SSH and in CI.
  if (!headless) {
    sim->RenderLoop();
  } else {
    while (!sim->exitrequest.load() && rclcpp::ok()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    sim->exitrequest.store(1);
  }
  active_sim.store(nullptr);
  physicsthreadhandle.join();

  // Release the embedded plugin and ROS nodes while the context is still
  // valid.  Keep the plugin objects alive until after the physics thread has
  // joined; the plugin owns the ControllerManager and its executor.  The
  // loader itself must outlive those objects, otherwise class_loader can try
  // to unload a library while controller instances are still on the heap.
  physics_plugins.clear();
  physics_plugin_loader.reset();
  if (d) {
    mj_deleteData(d);
    d = nullptr;
  }
  if (m) {
    mj_deleteModel(m);
    m = nullptr;
  }
  sim.reset();
  node.reset();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
