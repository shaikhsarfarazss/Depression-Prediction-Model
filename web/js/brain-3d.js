/**
 * NeuroPulse 3D Interactive WebGL Brain & Neural Synapse Engine
 * Procedural Dual-Hemisphere Cortex, Cerebellum & Brainstem
 * Surface-Pinned 10-20 Electrodes, Curved Axon Tractography & Action Potentials
 * Built with Three.js r128 + Selective UnrealBloom Post-Processing
 */

export function initNeuroBrain3D(containerId, onElectrodeHover = null) {
  const container = document.getElementById(containerId);
  const THREE = window.THREE;
  if (!container || !THREE) return null;

  // Clear existing canvas
  const existingCanvas = container.querySelector("canvas");
  if (existingCanvas) existingCanvas.remove();

  // 1. Scene, Camera, Renderer
  const scene = new THREE.Scene();
  const width = container.clientWidth || 700;
  const height = container.clientHeight || 500;

  const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
  camera.position.set(0, 2.5, 13.5);

  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    powerPreference: "high-performance"
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  container.appendChild(renderer.domElement);

  // Optional Postprocessing (UnrealBloom)
  let composer = null;
  if (THREE.EffectComposer && THREE.RenderPass && THREE.UnrealBloomPass) {
    try {
      composer = new THREE.EffectComposer(renderer);
      const renderPass = new THREE.RenderPass(scene, camera);
      composer.addPass(renderPass);

      const bloomPass = new THREE.UnrealBloomPass(
        new THREE.Vector2(width, height),
        0.75, // strength
        0.4,  // radius
        0.82  // threshold
      );
      composer.addPass(bloomPass);
    } catch (e) {
      console.warn("Postprocessing initialization skipped:", e);
      composer = null;
    }
  }

  // 2. Orbit Controls
  let controls = null;
  if (THREE.OrbitControls) {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.6;
    controls.enablePan = false;
    controls.minDistance = 6.5;
    controls.maxDistance = 20;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.75;
  }

  // 3. Clinical Cyber-Neuro Lighting
  const ambientLight = new THREE.AmbientLight(0x0a1020, 1.8);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0x00f0ff, 2.8);
  keyLight.position.set(7, 9, 8);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x8b5cf6, 2.4);
  fillLight.position.set(-8, -4, 6);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0x00f59b, 2.2);
  rimLight.position.set(0, -8, -8);
  scene.add(rimLight);

  const pointGlow = new THREE.PointLight(0x00f0ff, 2.2, 16);
  pointGlow.position.set(0, 0, 0);
  scene.add(pointGlow);

  // Root Master Group
  const brainRoot = new THREE.Group();
  scene.add(brainRoot);

  // 4. Procedural Dual-Hemisphere Cerebral Cortex Geometry
  function createHemisphere(isRight = false) {
    const group = new THREE.Group();
    const sign = isRight ? 1 : -1;

    const sphereGeo = new THREE.SphereGeometry(3.2, 48, 48);
    const pos = sphereGeo.attributes.position;

    for (let i = 0; i < pos.count; i++) {
      let x = pos.getX(i);
      let y = pos.getY(i);
      let z = pos.getZ(i);

      // Anatomical cerebral morphing
      x *= 0.78;
      y *= 0.85;
      z *= 1.25;

      // Temporal lobe depression & frontal elevation
      if (y < 0 && z > 0) y *= 0.9;
      if (z > 1.0) y += 0.22;

      // Longitudinal sagittal fissure separation
      x += sign * 0.45;

      // High-frequency harmonic gyri and sulci convolutions
      const freq1 = Math.sin(x * 4.2) * Math.cos(y * 4.5) * Math.sin(z * 4.0);
      const freq2 = Math.sin(x * 7.5 + y * 6.0) * 0.35;
      const convolution = (freq1 + freq2) * 0.16;

      pos.setXYZ(i, x + convolution, y + convolution, z + convolution);
    }
    sphereGeo.computeVertexNormals();

    const cortexMat = new THREE.MeshPhysicalMaterial({
      color: isRight ? 0x00f0ff : 0x8b5cf6,
      emissive: isRight ? 0x002244 : 0x220044,
      emissiveIntensity: 0.45,
      roughness: 0.22,
      metalness: 0.12,
      clearcoat: 0.9,
      clearcoatRoughness: 0.2,
      transmission: 0.68,
      thickness: 1.3,
      ior: 1.45,
      transparent: true,
      opacity: 0.82
    });

    const cortexMesh = new THREE.Mesh(sphereGeo, cortexMat);
    group.add(cortexMesh);

    // Subtle neon wireframe grid overlay
    const wireMat = new THREE.MeshBasicMaterial({
      color: isRight ? 0x00f0ff : 0xa78bfa,
      wireframe: true,
      transparent: true,
      opacity: 0.10
    });
    const wireMesh = new THREE.Mesh(sphereGeo, wireMat);
    wireMesh.scale.set(1.002, 1.002, 1.002);
    group.add(wireMesh);

    return { group, cortexMat, wireMat, mesh: cortexMesh };
  }

  const leftHemi = createHemisphere(false);
  const rightHemi = createHemisphere(true);
  brainRoot.add(leftHemi.group);
  brainRoot.add(rightHemi.group);

  // 5. Cerebellum (Two Posterior-Inferior Lobes) & Brainstem
  function createCerebellum() {
    const group = new THREE.Group();
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x6366f1,
      emissive: 0x1e1b4b,
      emissiveIntensity: 0.4,
      roughness: 0.3,
      transmission: 0.5,
      transparent: true,
      opacity: 0.75
    });

    // Left and right cerebellum lobes
    [-0.85, 0.85].forEach(xOff => {
      const cGeo = new THREE.SphereGeometry(1.25, 24, 24);
      const cPos = cGeo.attributes.position;
      for (let i = 0; i < cPos.count; i++) {
        let x = cPos.getX(i) * 0.9;
        let y = cPos.getY(i) * 0.65;
        let z = cPos.getZ(i) * 0.85;
        // Folia convolutions
        const folia = Math.sin(y * 14.0) * 0.05;
        cPos.setXYZ(i, x + folia, y, z + folia);
      }
      cGeo.computeVertexNormals();
      const cMesh = new THREE.Mesh(cGeo, mat);
      cMesh.position.set(xOff, -1.8, -1.7);
      group.add(cMesh);
    });

    // Brainstem (Pons & Medulla stalk)
    const stemGeo = new THREE.CylinderGeometry(0.55, 0.42, 2.2, 20);
    const stemMat = new THREE.MeshPhysicalMaterial({
      color: 0x00f0ff,
      emissive: 0x001f3f,
      emissiveIntensity: 0.35,
      roughness: 0.25,
      transmission: 0.6,
      transparent: true,
      opacity: 0.7
    });
    const stemMesh = new THREE.Mesh(stemGeo, stemMat);
    stemMesh.position.set(0, -2.4, -0.4);
    stemMesh.rotation.x = 0.2;
    group.add(stemMesh);

    return { group, mat, stemMat };
  }

  const subCortical = createCerebellum();
  brainRoot.add(subCortical.group);

  // 6. Neural Synapse Network & Curved Axon Tractography (CatmullRomCurves)
  const neuronCount = 280;
  const neuronPositions = [];
  const neuronColors = [];
  const colorCyan = new THREE.Color(0x00f0ff);
  const colorViolet = new THREE.Color(0x8b5cf6);
  const colorEmerald = new THREE.Color(0x00f59b);

  for (let i = 0; i < neuronCount; i++) {
    const u = Math.random();
    const v = Math.random();
    const theta = u * 2.0 * Math.PI;
    const phi = Math.acos(2.0 * v - 1.0);
    const r = 2.0 + Math.random() * 1.3;

    const x = r * Math.sin(phi) * Math.cos(theta) * 0.78 + (Math.random() > 0.5 ? 0.4 : -0.4);
    const y = r * Math.sin(phi) * Math.sin(theta) * 0.82;
    const z = r * Math.cos(phi) * 1.2;

    neuronPositions.push(new THREE.Vector3(x, y, z));
    const col = Math.random() > 0.5 ? colorCyan : (Math.random() > 0.5 ? colorViolet : colorEmerald);
    neuronColors.push(col.r, col.g, col.b);
  }

  // Curved Axons via CatmullRomCurve3
  const axonCurves = [];
  const axonPointsArr = [];

  for (let i = 0; i < neuronCount; i++) {
    for (let j = i + 1; j < neuronCount; j++) {
      const dist = neuronPositions[i].distanceTo(neuronPositions[j]);
      if (dist < 1.35 && Math.random() > 0.55) {
        const p1 = neuronPositions[i];
        const p2 = neuronPositions[j];
        // Midpoint arch for organic curvature
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        mid.y += (Math.random() - 0.5) * 0.35;
        mid.x *= 1.05;

        const curve = new THREE.CatmullRomCurve3([p1, mid, p2]);
        axonCurves.push(curve);

        const pts = curve.getPoints(12);
        for (let k = 0; k < pts.length - 1; k++) {
          axonPointsArr.push(pts[k], pts[k + 1]);
        }
      }
    }
  }

  const axonGeo = new THREE.BufferGeometry().setFromPoints(axonPointsArr);
  const axonMat = new THREE.LineBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.18,
    blending: THREE.AdditiveBlending
  });
  const axonLines = new THREE.LineSegments(axonGeo, axonMat);
  brainRoot.add(axonLines);

  // Neuron Soma Particles
  const somaGeo = new THREE.BufferGeometry();
  const somaPosArr = new Float32Array(neuronCount * 3);
  for (let i = 0; i < neuronCount; i++) {
    somaPosArr[i * 3 + 0] = neuronPositions[i].x;
    somaPosArr[i * 3 + 1] = neuronPositions[i].y;
    somaPosArr[i * 3 + 2] = neuronPositions[i].z;
  }
  somaGeo.setAttribute("position", new THREE.BufferAttribute(somaPosArr, 3));
  somaGeo.setAttribute("color", new THREE.Float32BufferAttribute(neuronColors, 3));

  const somaMat = new THREE.PointsMaterial({
    size: 0.18,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending
  });
  const somaPoints = new THREE.Points(somaGeo, somaMat);
  brainRoot.add(somaPoints);

  // Action Potential Sparks Traveling Along Actual Axon Curves
  const sparkCount = 50;
  const sparks = [];
  const sparkGeo = new THREE.SphereGeometry(0.065, 8, 8);
  const sparkMat = new THREE.MeshBasicMaterial({ color: 0xffffff, blending: THREE.AdditiveBlending });

  for (let s = 0; s < sparkCount; s++) {
    const curve = axonCurves.length > 0 ? axonCurves[Math.floor(Math.random() * axonCurves.length)] : null;
    const mesh = new THREE.Mesh(sparkGeo, sparkMat);
    brainRoot.add(mesh);
    sparks.push({
      curve: curve,
      progress: Math.random(),
      speed: 0.007 + Math.random() * 0.014,
      mesh: mesh
    });
  }

  // 7. Surface-Pinned 10-20 Standard EEG Electrodes
  const ELECTRODE_3D_COORDS = [
    { id: "Fp1", name: "Left Frontopolar",  pos: [-0.9, 2.0, 3.25],  region: "Frontopolar", tar: 0.82 },
    { id: "Fp2", name: "Right Frontopolar", pos: [ 0.9, 2.0, 3.25],  region: "Frontopolar", tar: 0.80 },
    { id: "F7",  name: "Left Anterior Temp",pos: [-2.85, 0.9, 1.8],  region: "Anterior Temporal", tar: 0.88 },
    { id: "F3",  name: "Left Frontal",      pos: [-1.6, 2.45, 1.85], region: "Frontal Cortex", tar: 0.85 },
    { id: "Fz",  name: "Midline Frontal",   pos: [ 0.0, 2.85, 1.95], region: "Midline Frontal", tar: 0.84 },
    { id: "F4",  name: "Right Frontal",     pos: [ 1.6, 2.45, 1.85], region: "Frontal Cortex", tar: 0.83 },
    { id: "F8",  name: "Right Anterior Temp",pos: [ 2.85, 0.9, 1.8], region: "Anterior Temporal", tar: 0.86 },
    { id: "T3",  name: "Left Mid-Temporal", pos: [-3.25, 0.2, 0.0],  region: "Mid-Temporal", tar: 0.90 },
    { id: "C3",  name: "Left Central",      pos: [-1.85, 2.75, 0.0], region: "Motor / Central", tar: 0.81 },
    { id: "Cz",  name: "Vertex / Central",  pos: [ 0.0, 3.25, 0.0],  region: "Vertex / Central", tar: 0.78 },
    { id: "C4",  name: "Right Central",     pos: [ 1.85, 2.75, 0.0], region: "Motor / Central", tar: 0.80 },
    { id: "T4",  name: "Right Mid-Temporal",pos: [ 3.25, 0.2, 0.0],  region: "Mid-Temporal", tar: 0.89 },
    { id: "T5",  name: "Left Post-Temporal",pos: [-2.75, 0.5, -2.05],region: "Posterior Temporal", tar: 0.92 },
    { id: "P3",  name: "Left Parietal",     pos: [-1.65, 2.25, -1.95],region: "Parietal Cortex", tar: 0.75 },
    { id: "Pz",  name: "Midline Parietal",  pos: [ 0.0, 2.85, -1.95],region: "Midline Parietal", tar: 0.74 },
    { id: "P4",  name: "Right Parietal",    pos: [ 1.65, 2.25, -1.95],region: "Parietal Cortex", tar: 0.76 },
    { id: "T6",  name: "Right Post-Temporal",pos: [ 2.75, 0.5, -2.05],region: "Posterior Temporal", tar: 0.91 },
    { id: "O1",  name: "Left Occipital",    pos: [-1.0, 1.25, -3.45],region: "Occipital Visual", tar: 0.65 },
    { id: "O2",  name: "Right Occipital",   pos: [ 1.0, 1.25, -3.45],region: "Occipital Visual", tar: 0.64 }
  ];

  const electrodeMeshes = [];
  const electrodeGroup = new THREE.Group();
  brainRoot.add(electrodeGroup);

  const elecGeo = new THREE.SphereGeometry(0.14, 16, 16);
  const ringGeo = new THREE.RingGeometry(0.18, 0.26, 24);

  ELECTRODE_3D_COORDS.forEach(elec => {
    const elecMat = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      emissive: 0x00f0ff,
      emissiveIntensity: 0.9,
      roughness: 0.2,
      metalness: 0.8
    });
    const mesh = new THREE.Mesh(elecGeo, elecMat);
    mesh.position.set(...elec.pos);
    mesh.userData = elec;

    // Glowing Halo Ring
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.lookAt(0, 0, 0);
    mesh.add(ringMesh);

    electrodeGroup.add(mesh);
    electrodeMeshes.push({ mesh, ringMesh, elecMat, ringMat, data: elec });
  });

  // 8. Dynamic State Management & Condition Heat-Mapping
  let currentMode = "normal";
  let pulseIntensity = 0.0;
  let animTime = 0;
  let isExploded = false;
  let targetHemiOffset = 0.0;
  let currentHemiOffset = 0.0;
  let isXRay = false;

  function setBrainMode(mode) {
    currentMode = mode.toLowerCase();
    const modeLabel = document.getElementById("active-brain-mode-label");

    if (currentMode === "alzheimer") {
      if (modeLabel) modeLabel.textContent = "MODE: ALZHEIMER'S (CORTICAL SLOWING)";
      leftHemi.cortexMat.color.setHex(0xff2a5f);
      rightHemi.cortexMat.color.setHex(0xff6b4a);
      leftHemi.cortexMat.emissive.setHex(0x550011);
      rightHemi.cortexMat.emissive.setHex(0x551100);
      axonMat.color.setHex(0xff6b4a);
      pointGlow.color.setHex(0xff2a5f);

      electrodeMeshes.forEach(em => {
        const isOccipital = em.data.id === "O1" || em.data.id === "O2";
        const col = isOccipital ? 0x6366f1 : 0xff2a5f;
        em.elecMat.color.setHex(col);
        em.elecMat.emissive.setHex(col);
        em.ringMat.color.setHex(col);
      });
    } else if (currentMode === "depression") {
      if (modeLabel) modeLabel.textContent = "MODE: DEPRESSION (FRONTAL ASYMMETRY)";
      leftHemi.cortexMat.color.setHex(0x4f46e5);
      rightHemi.cortexMat.color.setHex(0xffb703);
      leftHemi.cortexMat.emissive.setHex(0x1e1b4b);
      rightHemi.cortexMat.emissive.setHex(0x78350f);
      axonMat.color.setHex(0xffb703);
      pointGlow.color.setHex(0xffb703);

      electrodeMeshes.forEach(em => {
        const isLeftFrontal = em.data.id === "F3" || em.data.id === "Fp1" || em.data.id === "F7";
        const isRightFrontal = em.data.id === "F4" || em.data.id === "Fp2" || em.data.id === "F8";
        let col = 0x00f0ff;
        if (isLeftFrontal) col = 0x4f46e5; // Left hypoactivity
        else if (isRightFrontal) col = 0xffb703; // Right alpha excess
        em.elecMat.color.setHex(col);
        em.elecMat.emissive.setHex(col);
        em.ringMat.color.setHex(col);
      });
    } else {
      if (modeLabel) modeLabel.textContent = "MODE: HEALTHY BASELINE";
      leftHemi.cortexMat.color.setHex(0x8b5cf6);
      rightHemi.cortexMat.color.setHex(0x00f0ff);
      leftHemi.cortexMat.emissive.setHex(0x220044);
      rightHemi.cortexMat.emissive.setHex(0x003355);
      axonMat.color.setHex(0x00f0ff);
      pointGlow.color.setHex(0x00f0ff);

      electrodeMeshes.forEach(em => {
        const isOccipital = ["O1", "O2", "P3", "P4", "Pz"].includes(em.data.id);
        const col = isOccipital ? 0x00f59b : 0x00f0ff;
        em.elecMat.color.setHex(col);
        em.elecMat.emissive.setHex(col);
        em.ringMat.color.setHex(col);
      });
    }
  }

  function triggerNeuralPulse() {
    pulseIntensity = 1.0;
  }

  function toggleExplodeHemispheres() {
    isExploded = !isExploded;
    targetHemiOffset = isExploded ? 1.45 : 0.0;
    return isExploded;
  }

  function toggleXRay() {
    isXRay = !isXRay;
    leftHemi.cortexMat.opacity = isXRay ? 0.18 : 0.82;
    rightHemi.cortexMat.opacity = isXRay ? 0.18 : 0.82;
    leftHemi.wireMat.opacity = isXRay ? 0.45 : 0.10;
    rightHemi.wireMat.opacity = isXRay ? 0.45 : 0.10;
    subCortical.mat.opacity = isXRay ? 0.20 : 0.75;
    axonMat.opacity = isXRay ? 0.70 : 0.18;
    return isXRay;
  }

  // 9. Interactive Raycasting Tooltip on Electrode Hover
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const tooltip = document.getElementById("electrode-tooltip");
  const ttId = document.getElementById("tt-id");
  const ttName = document.getElementById("tt-name");
  const ttRegion = document.getElementById("tt-region");
  const ttPower = document.getElementById("tt-power");
  const ttTar = document.getElementById("tt-tar");
  let hoveredElecId = null;

  function onPointerMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const targets = electrodeMeshes.map(e => e.mesh);
    const intersects = raycaster.intersectObjects(targets);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      const data = hit.userData;
      if (tooltip && data) {
        tooltip.hidden = false;
        tooltip.style.left = `${event.clientX - rect.left}px`;
        tooltip.style.top = `${event.clientY - rect.top}px`;
        if (ttId) ttId.textContent = data.id;
        if (ttName) ttName.textContent = data.name;
        if (ttRegion) ttRegion.textContent = data.region;

        let pwr = 12.4;
        let localTarStr = "0.85 (Optimal)";
        if (currentMode === "alzheimer") {
          pwr = data.id.startsWith("O") ? 3.8 : 28.5;
          localTarStr = "3.48 (Pathological)";
        } else if (currentMode === "depression") {
          pwr = (data.id === "F3" || data.id === "Fp1") ? 22.8 : (data.id === "F4" ? 6.2 : 11.2);
          localTarStr = "1.65 (Frontal Asymm)";
        } else {
          pwr = (data.id.startsWith("O") || data.id.startsWith("P")) ? 19.4 : 8.5;
          localTarStr = "0.82 (Normative)";
        }

        if (ttPower) ttPower.textContent = `${pwr.toFixed(1)} μV²/Hz`;
        if (ttTar) ttTar.textContent = localTarStr;

        if (hoveredElecId !== data.id) {
          hoveredElecId = data.id;
          if (typeof onElectrodeHover === "function") {
            onElectrodeHover(data.id);
          }
        }
      }
    } else {
      if (tooltip) tooltip.hidden = true;
      if (hoveredElecId !== null) {
        hoveredElecId = null;
        if (typeof onElectrodeHover === "function") {
          onElectrodeHover(null);
        }
      }
    }
  }

  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", () => {
    if (tooltip) tooltip.hidden = true;
    if (typeof onElectrodeHover === "function") onElectrodeHover(null);
  });

  // 10. Spatial Orientation Compass Gizmo Update
  const gizmoCore = document.getElementById("gizmo-core");
  const gizmoAngleTxt = document.getElementById("gizmo-cam-angle");

  // 11. Animation Loop
  let isRunning = true;
  function animate() {
    if (!isRunning) return;
    requestAnimationFrame(animate);

    animTime += 0.018;

    // Smooth hemisphere explosion animation
    currentHemiOffset += (targetHemiOffset - currentHemiOffset) * 0.1;
    leftHemi.group.position.x = -currentHemiOffset;
    rightHemi.group.position.x = currentHemiOffset;

    if (controls) {
      controls.update();

      // Update orientation gizmo
      if (gizmoCore && controls.getAzimuthalAngle) {
        const rad = controls.getAzimuthalAngle();
        const deg = Math.round(rad * (180 / Math.PI));
        gizmoCore.style.transform = `rotate(${-deg}deg)`;
        if (gizmoAngleTxt) gizmoAngleTxt.textContent = `AZIMUTH: ${deg > 0 ? deg : 360 + deg}°`;
      }
    }

    // Dynamic rhythm breathing
    let freqRate = 1.8;
    let sparkSpeedMult = 1.0;
    if (currentMode === "alzheimer") {
      freqRate = 0.8;
      sparkSpeedMult = 0.45; // Sluggish propagation in Alzheimer's
    } else if (currentMode === "depression") {
      freqRate = 1.4;
      sparkSpeedMult = 0.9;
    }

    const wavePulse = Math.sin(animTime * freqRate) * 0.035;
    const pulseBoost = pulseIntensity * 0.14;
    brainRoot.scale.set(1.0 + wavePulse + pulseBoost, 1.0 + wavePulse + pulseBoost, 1.0 + wavePulse + pulseBoost);

    if (pulseIntensity > 0) pulseIntensity *= 0.93;

    // Action potential sparks along axon curves
    sparks.forEach(sp => {
      sp.progress += sp.speed * sparkSpeedMult;
      if (sp.progress >= 1.0) {
        sp.progress = 0;
        if (axonCurves.length > 0) {
          sp.curve = axonCurves[Math.floor(Math.random() * axonCurves.length)];
        }
      }
      if (sp.curve) {
        const pt = sp.curve.getPoint(sp.progress);
        sp.mesh.position.copy(pt);
      }
    });

    // Electrode halo pulsation
    electrodeMeshes.forEach((em, idx) => {
      const ringScale = 1.0 + Math.sin(animTime * 3.0 + idx * 0.5) * 0.35 + pulseIntensity;
      em.ringMesh.scale.set(ringScale, ringScale, 1.0);
    });

    // Render with postprocessing bloom if available
    if (composer) {
      composer.render();
    } else {
      renderer.render(scene, camera);
    }
  }

  animate();

  function onResize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    if (composer) composer.setSize(w, h);
  }
  window.addEventListener("resize", onResize);

  return {
    setBrainMode,
    triggerNeuralPulse,
    toggleExplodeHemispheres,
    toggleXRay,
    toggleAutoRotate: (val) => { if (controls) controls.autoRotate = val; },
    resetCamera: () => {
      camera.position.set(0, 2.5, 13.5);
      if (controls) controls.target.set(0, 0, 0);
    },
    destroy: () => {
      isRunning = false;
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.dispose();
      container.innerHTML = "";
    }
  };
}
