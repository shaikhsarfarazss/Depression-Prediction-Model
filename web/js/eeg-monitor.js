/**
 * NeuroPulse Real-Time 19-Channel Digital EEG Oscilloscope Monitor
 * Zero-ghosting phosphor circular buffer sweep with dynamic timebase & gain
 * Synchronized with 3D brain electrode raycasting & 2D scalp topography
 */

export const STANDARD_CHANNELS = [
  "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
  "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
  "Pz", "P4", "T6", "O1", "O2"
];

export function initEEGMonitor(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");

  let width = (canvas.width = canvas.parentElement.clientWidth || 800);
  let height = (canvas.height = canvas.parentElement.clientHeight || 400);

  let isPlaying = true;
  let currentCondition = "normal";
  let showClean = true;
  let sweepSpeed = 2.0; // Corresponds to ~25 mm/s
  let gain = 1.0; // 10 μV standard
  let sweepX = 0;
  let highlightedChannel = null;

  let channelCount = 8;
  let activeChannels = STANDARD_CHANNELS.slice(0, channelCount);

  // History buffer
  let bufferLength = width;
  let channelBuffers = activeChannels.map(() => new Float32Array(bufferLength));

  let timeSample = 0;

  function generateSample(chName, t, condition, clean) {
    const freq_delta = 2.0;
    const freq_theta = 6.0;
    const freq_alpha = 10.2;
    const freq_beta  = 20.0;
    const freq_gamma = 36.0;

    const delta = Math.sin(t * freq_delta * 2 * Math.PI);
    const theta = Math.sin(t * freq_theta * 2 * Math.PI + 0.4);
    const alpha = Math.sin(t * freq_alpha * 2 * Math.PI + 0.8);
    const beta  = Math.sin(t * freq_beta  * 2 * Math.PI + 1.2);
    const gamma = Math.sin(t * freq_gamma * 2 * Math.PI + 0.5);

    let val = 0;
    if (condition === "normal") {
      const isOccipital = ["O1", "O2", "P3", "P4", "Pz"].includes(chName);
      val = (isOccipital ? 20.0 : 8.0) * alpha + 6.0 * beta + 3.0 * theta + 2.0 * delta + 1.2 * gamma;
    } else if (condition === "alzheimer") {
      // Pathological EEG slowing: dominant slow delta and theta
      val = 24.0 * delta + 20.0 * theta + 2.0 * alpha + 1.5 * beta;
    } else if (condition === "depression") {
      // Frontal alpha asymmetry (FAA)
      let alphaAmp = 7.0;
      if (chName === "F3" || chName === "Fp1") alphaAmp = 17.5;
      else if (chName === "F4" || chName === "Fp2") alphaAmp = 5.0;
      val = alphaAmp * alpha + 13.0 * theta + 4.0 * beta + 3.0 * delta;
    }

    if (!clean) {
      val += (Math.random() - 0.5) * 8.5; // background noise
      val += Math.sin(t * 50.0 * 2 * Math.PI) * 5.0; // 50 Hz power-line hum
      // Blink artifact on Fp1/Fp2 every 3 seconds
      if ((chName === "Fp1" || chName === "Fp2") && (t % 3.0 < 0.25)) {
        val += 60.0 * Math.sin((t % 3.0) / 0.25 * Math.PI);
      }
    }

    return val;
  }

  function resize() {
    if (!canvas.parentElement) return;
    width = canvas.width = canvas.parentElement.clientWidth || 800;
    height = canvas.height = canvas.parentElement.clientHeight || 400;
    bufferLength = width;
    channelBuffers = activeChannels.map(() => new Float32Array(bufferLength));
  }
  window.addEventListener("resize", resize);

  function render() {
    if (!isPlaying) {
      requestAnimationFrame(render);
      return;
    }

    const dt = (1.0 / 60.0) * sweepSpeed;
    timeSample += dt;

    // Advance sweep
    const step = 2.4 * sweepSpeed;
    const prevSweepX = sweepX;
    sweepX = (sweepX + step) % width;

    // Clear sweep beam margin ahead with slight phosphor persistence
    const clearWidth = 32;
    ctx.fillStyle = "rgba(6, 9, 17, 0.42)";
    ctx.fillRect(sweepX, 0, clearWidth, height);

    const rowHeight = height / (channelCount + 1);

    // Draw EEG channel baselines and traces
    activeChannels.forEach((ch, chIdx) => {
      const baseY = (chIdx + 1) * rowHeight;
      const isHighlighted = highlightedChannel === ch;

      // Draw baseline grid line
      ctx.strokeStyle = isHighlighted ? "rgba(0, 240, 255, 0.35)" : "rgba(255, 255, 255, 0.04)";
      ctx.lineWidth = isHighlighted ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(0, baseY);
      ctx.lineTo(width, baseY);
      ctx.stroke();

      // Sample new voltage
      const rawVal = generateSample(ch, timeSample, currentCondition, showClean);
      const scaledVal = rawVal * gain * 0.42;

      const currentPixelX = Math.floor(sweepX);
      if (channelBuffers[chIdx]) {
        channelBuffers[chIdx][currentPixelX] = scaledVal;
      }

      // Channel label
      ctx.fillStyle = isHighlighted ? "#FFFFFF" : (showClean ? "#00F0FF" : "#FF5252");
      ctx.font = channelCount === 19 ? "bold 9.5px 'JetBrains Mono', monospace" : "bold 11.5px 'JetBrains Mono', monospace";
      ctx.fillText(ch, 10, baseY - (channelCount === 19 ? 3 : 6));

      // Trace line segment
      let strokeColor = showClean ? "#00F59B" : "#FF7B72";
      if (isHighlighted) strokeColor = "#00F0FF";

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = isHighlighted ? 2.0 : (channelCount === 19 ? 1.0 : 1.35);
      if (isHighlighted) {
        ctx.shadowColor = "#00F0FF";
        ctx.shadowBlur = 8;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      const prevX = (prevSweepX + width) % width;
      const prevY = baseY - ((channelBuffers[chIdx] && channelBuffers[chIdx][Math.floor(prevX)]) || 0);
      const currY = baseY - scaledVal;

      if (prevSweepX < sweepX) {
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(sweepX, currY);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
    });

    // Glowing phosphor sweep beam line
    ctx.strokeStyle = "#00F0FF";
    ctx.lineWidth = 1.8;
    ctx.shadowColor = "#00F0FF";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(sweepX, 0);
    ctx.lineTo(sweepX, height);
    ctx.stroke();
    ctx.shadowBlur = 0;

    requestAnimationFrame(render);
  }

  render();

  return {
    setCondition: (cond) => { currentCondition = cond.toLowerCase(); },
    toggleFilter: () => { showClean = !showClean; return showClean; },
    toggleMontage: () => {
      channelCount = channelCount === 8 ? 19 : 8;
      activeChannels = STANDARD_CHANNELS.slice(0, channelCount);
      channelBuffers = activeChannels.map(() => new Float32Array(bufferLength));
      return channelCount;
    },
    cycleTimebase: () => {
      // 10 mm/s (1.0), 25 mm/s (2.0), 50 mm/s (3.5)
      const speeds = [1.0, 2.0, 3.5];
      const labels = ["10mm/s", "25mm/s", "50mm/s"];
      const idx = speeds.indexOf(sweepSpeed);
      const nextIdx = (idx + 1) % speeds.length;
      sweepSpeed = speeds[nextIdx];
      return labels[nextIdx];
    },
    cycleGain: () => {
      // 5 μV (0.5), 10 μV (1.0), 20 μV (1.8)
      const gains = [0.5, 1.0, 1.8];
      const labels = ["5μV", "10μV", "20μV"];
      const idx = gains.indexOf(gain);
      const nextIdx = (idx + 1) % gains.length;
      gain = gains[nextIdx];
      return labels[nextIdx];
    },
    togglePlay: () => {
      isPlaying = !isPlaying;
      return isPlaying;
    },
    highlightChannel: (chName) => {
      highlightedChannel = chName;
    },
    destroy: () => { window.removeEventListener("resize", resize); }
  };
}
