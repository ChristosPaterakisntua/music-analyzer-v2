# Preventing Google Colab Runtime Disconnections

Long training sessions on Google Colab (especially on the free tier) can terminate unexpectedly. Below are the four primary causes of runtime disconnections and how to mitigate them.

---

## Common Causes of Disconnection

1. **Browser Tab Throttling:** Modern browsers freeze inactive background tabs to save memory, severing Colab's WebSocket connection within 15–30 minutes.
2. **System Sleep & Network Drops:** Allowing your computer to enter sleep mode or changing IP addresses (e.g., via VPN server switching) immediately terminates the session.
3. **Idle Timeout (~90 Minutes):** Colab disconnects runtimes if no user interface activity is detected for roughly 90 minutes, even while code is executing.
4. **Resource & Quota Limits:** Exceeding System RAM (~12.7 GB on the free tier) crashes the runtime immediately, while hitting the dynamic daily GPU limit (~2–4 hours) forces a cooldown disconnect.

---

## Preventive Measures

### 1. Disable Browser Tab Sleeping
Prevent your browser from suspending the Colab tab:
1. Open your browser's performance settings:
   - **Chrome:** `chrome://settings/performance`
   - **Edge:** `edge://settings/system`
2. Under **General** → *Always keep these sites active*, click **Add** and enter:
   ```text
   colab.research.google.com
   ```
3. Keep Colab open in a **dedicated, non-minimized window** rather than a hidden background tab.

---

### 2. Configure System Power & Network Settings
Ensure your local machine maintains an uninterrupted connection:
1. Keep your laptop plugged into AC power during training.
2. Disable automatic sleep when plugged in (**Settings → System → Power & battery → Screen and sleep** → set *Put my device to sleep after* to **Never**; allowing only the display to turn off is safe).
3. Ensure any VPN connection is stable and does not automatically switch servers mid-session.

---

### 3. Run a Keep-Alive Script in the Browser Console
To prevent the ~90-minute UI idle timeout, periodically trigger a lightweight click event:
1. On the Colab page, open Developer Tools Console (`Ctrl + Shift + J` on Windows/Linux, `Cmd + Option + J` on macOS, or `F12` → **Console**).
2. If prompted by the browser, type `allow pasting` and press `Enter`.
3. Paste and run the following snippet:
   ```javascript
   function KeepClicking() {
       console.log("Colab keep-alive ping:", new Date().toLocaleTimeString());
       const connectBtn = document.querySelector("colab-connect-button");
       if (connectBtn && connectBtn.shadowRoot) {
           const btn = connectBtn.shadowRoot.querySelector("#connect");
           if (btn) btn.click();
       }
   }
   const keepAliveInterval = setInterval(KeepClicking, 60000);
   ```

> **Note:** This prevents idle timeouts, though it cannot bypass daily GPU usage quotas or interactive CAPTCHA prompts.

---

### Checkpoint Recovery 
Because training checkpoints are saved automatically to Google Drive (`CHECKPOINT_DIR`), resume interrupted runs instead of restarting from epoch 1.