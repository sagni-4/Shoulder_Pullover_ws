#include "autoware_shoulder_pullover_manager/keyboard_trigger.hpp"

#include <poll.h>
#include <unistd.h>

#include <cctype>

namespace autoware::shoulder_pullover_manager
{

KeyboardTrigger::KeyboardTrigger(char trigger_key, std::function<void()> on_trigger)
: trigger_key_(static_cast<char>(std::tolower(static_cast<unsigned char>(trigger_key)))),
  on_trigger_(std::move(on_trigger))
{
}

KeyboardTrigger::~KeyboardTrigger() { stop(); }

void KeyboardTrigger::start()
{
  if (running_.exchange(true)) {
    return;  // Already started.
  }

  if (::isatty(STDIN_FILENO) == 0) {
    // No attached terminal (e.g. running as a background/composable node,
    // or with stdin redirected) -- skip raw-mode entirely rather than
    // failing. The OperateMrm service remains the fully-functional trigger
    // path in this case.
    active_ = false;
    running_ = false;
    return;
  }

  if (::tcgetattr(STDIN_FILENO, &original_termios_) != 0) {
    active_ = false;
    running_ = false;
    return;
  }
  termios_saved_ = true;

  struct termios raw = original_termios_;
  raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
  raw.c_cc[VMIN] = 0;   // Non-blocking-ish: read() may return 0 bytes.
  raw.c_cc[VTIME] = 0;  // No inter-byte timeout; poll() governs blocking instead.
  ::tcsetattr(STDIN_FILENO, TCSANOW, &raw);

  active_ = true;
  thread_ = std::thread(&KeyboardTrigger::run, this);
}

void KeyboardTrigger::stop()
{
  const bool was_running = running_.exchange(false);
  if (was_running && thread_.joinable()) {
    thread_.join();
  }
  if (termios_saved_) {
    ::tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
    termios_saved_ = false;
  }
}

void KeyboardTrigger::run()
{
  struct pollfd poll_fd
  {
    STDIN_FILENO, POLLIN, 0
  };

  // Poll with a short timeout purely so the running_ flag is re-checked
  // regularly, letting stop() return promptly instead of blocking on the
  // next keypress indefinitely.
  constexpr int kPollTimeoutMs = 200;

  while (running_) {
    const int ready = ::poll(&poll_fd, 1, kPollTimeoutMs);
    if (ready <= 0) {
      continue;  // Timeout or interrupted -- just re-check running_.
    }

    char c = 0;
    const ssize_t n = ::read(STDIN_FILENO, &c, 1);
    if (n != 1) {
      continue;
    }

    if (static_cast<char>(std::tolower(static_cast<unsigned char>(c))) == trigger_key_) {
      on_trigger_();
    }
  }
}

}  // namespace autoware::shoulder_pullover_manager
