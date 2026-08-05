#pragma once

#include <termios.h>

#include <atomic>
#include <functional>
#include <thread>

namespace autoware::shoulder_pullover_manager
{

/// Reads raw (non-canonical, no-echo) stdin on a background thread and
/// invokes a callback whenever the configured trigger key is pressed.
///
/// This exists purely to simulate an MRM trigger for development/demo
/// purposes -- the real trigger path is the `OperateMrm` service
/// (PullOverManagerNode::onOperateMrm), which this class does not replace or
/// bypass; both converge on the same internal maneuver logic.
///
/// Safety/engineering notes:
///  - The terminal is only ever put into raw mode if stdin is actually a
///    TTY (`isatty`); when launched as a composable/background node with no
///    attached terminal, this silently becomes a no-op rather than failing
///    or corrupting a pipe.
///  - The original termios state is saved and restored on destruction
///    (RAII), so a crash during normal C++ stack unwinding, or a clean
///    shutdown, never leaves the user's shell in raw mode.
///  - The callback is invoked directly on the background reader thread. It
///    must therefore be limited to something thread-safe and cheap (e.g.
///    setting an `std::atomic<bool>`) -- it must NOT call into rclcpp
///    publishers/services/clients directly. PullOverManagerNode follows
///    this contract by only flipping an atomic flag here and doing the
///    actual work from a wall timer on the executor thread.
class KeyboardTrigger
{
public:
  KeyboardTrigger(char trigger_key, std::function<void()> on_trigger);
  ~KeyboardTrigger();

  KeyboardTrigger(const KeyboardTrigger &) = delete;
  KeyboardTrigger & operator=(const KeyboardTrigger &) = delete;
  KeyboardTrigger(KeyboardTrigger &&) = delete;
  KeyboardTrigger & operator=(KeyboardTrigger &&) = delete;

  /// Starts the background reader thread. Safe to call at most once.
  void start();

  /// Stops the reader thread and restores the terminal, if it was changed.
  /// Safe to call multiple times (idempotent) and is also called from the
  /// destructor.
  void stop();

  /// True if raw-mode stdin reading is actually active (i.e. stdin was a
  /// TTY at start() time). Exposed mainly for startup logging.
  [[nodiscard]] bool isActive() const { return active_; }

private:
  void run();

  char trigger_key_;
  std::function<void()> on_trigger_;

  std::atomic<bool> running_{false};
  bool active_{false};
  std::thread thread_;

  struct termios original_termios_
  {
  };
  bool termios_saved_{false};
};

}  // namespace autoware::shoulder_pullover_manager
