import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

function material(color, options = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.76, ...options });
}

function addBox(group, size, position, boxMaterial, rotationY = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), boxMaterial);
  mesh.position.set(...position);
  mesh.rotation.y = rotationY;
  group.add(mesh);
  return mesh;
}

function addCylinder(group, radius, height, position, cylinderMaterial, segments = 14) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, height, segments),
    cylinderMaterial,
  );
  mesh.position.set(...position);
  group.add(mesh);
  return mesh;
}

function createManorRoom() {
  const room = new THREE.Group();
  const oak = material(0x3b241a, { roughness: 0.82 });
  const panel = material(0x241b19, { roughness: 0.9 });
  const stone = material(0x82705d, { roughness: 0.95 });
  const brass = material(0x8b6a34, { metalness: 0.55, roughness: 0.38 });

  addBox(room, [14, 0.2, 10], [0, -0.13, 0], oak);
  addBox(room, [14.2, 5.2, 0.24], [0, 2.5, -5.02], panel);
  addBox(room, [0.24, 5.2, 10.2], [-7.02, 2.5, 0], panel);

  for (let plank = -6; plank <= 6; plank += 1) {
    addBox(room, [0.025, 0.014, 9.8], [plank, -0.015, 0], material(0x25150f));
  }

  // Raised timber panels, skirting, and crown moulding give the room its manor scale.
  addBox(room, [14, 0.2, 0.18], [0, 0.18, -4.84], stone);
  addBox(room, [14, 0.16, 0.2], [0, 4.72, -4.82], stone);
  addBox(room, [0.18, 0.2, 10], [-6.84, 0.18, 0], stone);
  addBox(room, [0.18, 0.16, 10], [-6.82, 4.72, 0], stone);
  for (const x of [-6.1, -3.05, 3.05, 6.1]) {
    addBox(room, [0.1, 4.3, 0.12], [x, 2.38, -4.82], stone);
  }
  for (const z of [-4.1, -1.35, 1.35, 4.1]) {
    addBox(room, [0.12, 4.3, 0.1], [-6.82, 2.38, z], stone);
  }

  const nightGlass = material(0x10192b, {
    emissive: 0x0b1426,
    emissiveIntensity: 0.32,
    metalness: 0.05,
    roughness: 0.26,
  });
  const curtain = material(0x351c25, { roughness: 0.98 });
  for (const x of [-4.55, 4.55]) {
    addBox(room, [2.35, 2.75, 0.08], [x, 2.75, -4.84], nightGlass);
    addBox(room, [2.58, 0.13, 0.14], [x, 4.18, -4.74], stone);
    addBox(room, [2.58, 0.13, 0.14], [x, 1.32, -4.74], stone);
    addBox(room, [0.12, 2.92, 0.14], [x - 1.23, 2.75, -4.74], stone);
    addBox(room, [0.12, 2.92, 0.14], [x + 1.23, 2.75, -4.74], stone);
    addBox(room, [0.08, 2.75, 0.15], [x, 2.75, -4.72], stone);
    addBox(room, [0.42, 3.15, 0.22], [x - 1.42, 2.72, -4.65], curtain);
    addBox(room, [0.42, 3.15, 0.22], [x + 1.42, 2.72, -4.65], curtain);
    addCylinder(room, 0.06, 3.18, [x - 1.42, 2.72, -4.5], brass, 10);
  }
  return room;
}

function createFireplace() {
  const fireplace = new THREE.Group();
  const stone = material(0x8b7964, { roughness: 0.96 });
  const darkStone = material(0x2a211d, { roughness: 1 });
  const charredWood = material(0x28130d, { roughness: 1 });

  addBox(fireplace, [4.5, 0.26, 1.15], [0, 0.05, 0.42], stone);
  addBox(fireplace, [3.9, 3.05, 0.55], [0, 1.65, 0.17], stone);
  addBox(fireplace, [2.45, 1.75, 0.12], [0, 0.95, 0.5], darkStone);
  addBox(fireplace, [4.55, 0.32, 0.84], [0, 3.18, 0.3], stone);
  addBox(fireplace, [0.32, 3.25, 0.76], [-2.0, 1.62, 0.3], stone);
  addBox(fireplace, [0.32, 3.25, 0.76], [2.0, 1.62, 0.3], stone);

  for (const rotation of [-0.28, 0.28]) {
    const log = addCylinder(fireplace, 0.13, 1.65, [rotation * 1.5, 0.48, 0.64], charredWood, 10);
    log.rotation.z = Math.PI / 2;
    log.rotation.y = rotation;
  }

  const flames = [];
  const flameColors = [0xff5b18, 0xffa52e, 0xffd36a];
  for (let index = 0; index < 5; index += 1) {
    const flameMaterial = material(flameColors[index % flameColors.length], {
      emissive: flameColors[index % flameColors.length],
      emissiveIntensity: 1.35,
      roughness: 0.35,
    });
    const flame = new THREE.Mesh(new THREE.SphereGeometry(0.24, 10, 8), flameMaterial);
    flame.scale.set(0.72, 1.8 + (index % 2) * 0.65, 0.65);
    flame.position.set(-0.7 + index * 0.35, 0.62 + (index % 2) * 0.14, 0.72);
    fireplace.add(flame);
    flames.push(flame);
  }

  const fireLight = new THREE.PointLight(0xff7a2c, 13.0, 14, 1.65);
  fireLight.position.set(0, 1.1, 1.65);
  fireplace.add(fireLight);
  return { group: fireplace, fireLight, flames };
}

function createPortrait() {
  const portrait = new THREE.Group();
  const frame = material(0x8d6b34, { metalness: 0.42, roughness: 0.42 });
  addBox(portrait, [2.15, 1.55, 0.12], [0, 0, 0], frame);
  addBox(portrait, [1.82, 1.22, 0.08], [0, 0, 0.08], material(0x182029));
  addBox(portrait, [0.55, 0.78, 0.05], [0, -0.12, 0.14], material(0x3e4650));
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 12, 8), material(0x8f7968));
  head.position.set(0, 0.35, 0.16);
  portrait.add(head);
  return portrait;
}

function createSofa() {
  const sofa = new THREE.Group();
  const leather = material(0x3e2022, { roughness: 0.68 });
  const cushion = material(0x5a3030, { roughness: 0.8 });
  addBox(sofa, [5.2, 0.62, 1.35], [0, 0.54, 0], leather);
  addBox(sofa, [5.2, 1.52, 0.42], [0, 1.08, 0.48], leather);
  addBox(sofa, [0.46, 0.9, 1.56], [-2.48, 0.75, 0], leather);
  addBox(sofa, [0.46, 0.9, 1.56], [2.48, 0.75, 0], leather);
  for (const x of [-1.55, 0, 1.55]) {
    const seat = addBox(sofa, [1.36, 0.24, 1.05], [x, 0.9, -0.08], cushion);
    seat.rotation.x = -0.05;
  }
  return sofa;
}

function createChair() {
  const chair = new THREE.Group();
  const leather = material(0x56312a, { roughness: 0.76 });
  addBox(chair, [1.65, 0.55, 1.45], [0, 0.5, 0], leather);
  addBox(chair, [1.65, 1.45, 0.38], [0, 1.08, 0.5], leather);
  addBox(chair, [0.3, 0.9, 1.4], [-0.78, 0.75, 0], leather);
  addBox(chair, [0.3, 0.9, 1.4], [0.78, 0.75, 0], leather);
  return chair;
}

function createCoffeeTable() {
  const table = new THREE.Group();
  const carvedWood = material(0x3b2117, { roughness: 0.66 });
  addBox(table, [3.2, 0.22, 1.65], [0, 0.72, 0], carvedWood);
  for (const x of [-1.3, 1.3]) {
    for (const z of [-0.62, 0.62]) {
      addCylinder(table, 0.11, 0.72, [x, 0.34, z], carvedWood, 12);
    }
  }
  const book = addBox(table, [0.72, 0.08, 0.48], [0.65, 0.86, 0.05], material(0x69202c));
  book.rotation.y = -0.22;
  return table;
}

function createTelephoneTable() {
  const table = new THREE.Group();
  const carvedWood = material(0x321b13, { roughness: 0.7 });
  const brass = material(0x8c6a32, { metalness: 0.62, roughness: 0.34 });
  addBox(table, [1.3, 0.18, 1.05], [0, 0.72, 0], carvedWood);
  addBox(table, [1.08, 0.1, 0.83], [0, 0.2, 0], carvedWood);
  for (const x of [-0.5, 0.5]) {
    for (const z of [-0.38, 0.38]) {
      addCylinder(table, 0.075, 0.7, [x, 0.35, z], carvedWood, 12);
    }
  }
  addBox(table, [0.82, 0.025, 0.025], [0, 0.82, 0.52], brass);
  return table;
}

function createTelephone() {
  const group = new THREE.Group();
  const black = material(0x11100f, { metalness: 0.24, roughness: 0.34 });
  const brass = material(0x9b7438, { metalness: 0.72, roughness: 0.28 });
  const ivory = material(0xd9c9a6, { roughness: 0.68 });
  const indicator = material(0x551711, {
    emissive: 0xff3b24,
    emissiveIntensity: 0.08,
    roughness: 0.4,
  });

  addBox(group, [0.92, 0.2, 0.58], [0, 0.1, 0], black);
  addBox(group, [0.7, 0.18, 0.42], [0, 0.25, 0.03], black).rotation.x = -0.22;
  const dial = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.035, 8, 24), brass);
  dial.position.set(0, 0.31, 0.22);
  dial.rotation.x = -0.22;
  group.add(dial);
  const dialCenter = new THREE.Mesh(new THREE.CircleGeometry(0.07, 18), ivory);
  dialCenter.position.set(0, 0.31, 0.258);
  dialCenter.rotation.x = -0.22;
  group.add(dialCenter);

  const handset = new THREE.Group();
  addBox(handset, [0.72, 0.11, 0.14], [0, 0, 0], black);
  for (const x of [-0.39, 0.39]) {
    const receiver = addCylinder(handset, 0.12, 0.2, [x, 0, 0], black, 14);
    receiver.rotation.z = Math.PI / 2;
  }
  handset.position.set(0, 0.46, -0.02);
  group.add(handset);

  const bell = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 7), indicator);
  bell.position.set(0.33, 0.3, 0.24);
  group.add(bell);
  return { group, handset, indicator };
}

function createTurntableTable() {
  const table = new THREE.Group();
  const walnut = material(0x3a2117, { roughness: 0.68 });
  const brass = material(0x8c6a32, { metalness: 0.62, roughness: 0.34 });
  addBox(table, [1.55, 0.18, 1.22], [0, 0.72, 0], walnut);
  addBox(table, [1.34, 0.12, 1.02], [0, 0.22, 0], walnut);
  for (const x of [-0.61, 0.61]) {
    for (const z of [-0.45, 0.45]) {
      addCylinder(table, 0.075, 0.7, [x, 0.35, z], walnut, 12);
    }
  }
  addBox(table, [1.08, 0.025, 0.025], [0, 0.82, 0.6], brass);
  return table;
}

function createTurntable() {
  const group = new THREE.Group();
  const wood = material(0x4a281b, { roughness: 0.62 });
  const black = material(0x10100f, { metalness: 0.16, roughness: 0.42 });
  const vinyl = material(0x090909, { metalness: 0.25, roughness: 0.28 });
  const brass = material(0x9b7438, { metalness: 0.72, roughness: 0.25 });
  const label = material(0x6f2023, { roughness: 0.58 });

  addBox(group, [1.25, 0.16, 0.92], [0, 0.08, 0], wood);
  const platter = addCylinder(group, 0.39, 0.075, [-0.19, 0.2, 0], black, 40);
  const record = addCylinder(group, 0.34, 0.018, [-0.19, 0.25, 0], vinyl, 48);
  const recordLabel = addCylinder(group, 0.105, 0.022, [-0.19, 0.267, 0], label, 24);
  const spindle = addCylinder(group, 0.018, 0.055, [-0.19, 0.29, 0], brass, 12);
  void platter;
  void spindle;

  const tonearm = new THREE.Group();
  tonearm.position.set(0.43, 0.29, -0.28);
  addCylinder(tonearm, 0.055, 0.09, [0, 0, 0], brass, 14);
  const arm = addBox(tonearm, [0.055, 0.055, 0.58], [-0.18, 0.04, 0.23], brass);
  arm.rotation.y = -0.7;
  addBox(tonearm, [0.13, 0.07, 0.12], [-0.37, 0.04, 0.43], black, -0.7);
  group.add(tonearm);

  const indicator = material(0x152018, {
    emissive: 0x45d06f,
    emissiveIntensity: 0.05,
    roughness: 0.42,
  });
  addCylinder(group, 0.035, 0.018, [0.47, 0.18, 0.31], indicator, 12);
  return { group, record, recordLabel, tonearm, indicator };
}

function createBookcase() {
  const shelf = new THREE.Group();
  const frame = material(0x301c14, { roughness: 0.76 });
  addBox(shelf, [2.2, 3.55, 0.38], [0, 1.78, 0.2], frame);
  addBox(shelf, [1.88, 3.2, 0.42], [0, 1.76, -0.03], material(0x1c1513));
  for (const y of [0.72, 1.5, 2.28, 3.06]) {
    addBox(shelf, [2.05, 0.1, 0.56], [0, y, -0.08], frame);
  }
  const colors = [0x572832, 0x233b45, 0x6c4b27, 0x34442e, 0x49364d];
  for (let index = 0; index < 15; index += 1) {
    const row = index % 3;
    const column = Math.floor(index / 3);
    addBox(
      shelf,
      [0.24 + (index % 2) * 0.05, 0.5 + (index % 3) * 0.07, 0.25],
      [-0.72 + column * 0.36, 0.43 + row * 0.78, -0.36],
      material(colors[index % colors.length]),
    );
  }
  return shelf;
}

function createClimateSystem() {
  const climate = new THREE.Group();
  const brass = material(0x8c6a32, { metalness: 0.6, roughness: 0.38 });
  const darkMetal = material(0x211c1a, { metalness: 0.35, roughness: 0.55 });

  // A low brass register suggests concealed ducting without introducing a modern wall unit.
  addBox(climate, [2.45, 1.05, 0.16], [0, -1.25, 0], brass);
  addBox(climate, [2.12, 0.74, 0.08], [0, -1.25, 0.13], darkMetal);
  for (const y of [-1.48, -1.33, -1.18, -1.03]) {
    addBox(climate, [1.88, 0.055, 0.06], [0, y, 0.21], brass);
  }

  const dialFaceMaterial = material(0x4b6572, {
    emissive: 0x4b6572,
    emissiveIntensity: 0.45,
    roughness: 0.42,
  });
  const dialFace = new THREE.Mesh(new THREE.CircleGeometry(0.48, 24), dialFaceMaterial);
  dialFace.position.set(0, 0.58, 0.08);
  climate.add(dialFace);
  const dialRing = new THREE.Mesh(new THREE.TorusGeometry(0.53, 0.075, 10, 28), brass);
  dialRing.position.set(0, 0.58, 0.14);
  climate.add(dialRing);

  const needlePivot = new THREE.Group();
  needlePivot.position.set(0, 0.58, 0.2);
  addBox(needlePivot, [0.055, 0.39, 0.055], [0, 0.18, 0], brass);
  climate.add(needlePivot);
  for (let index = 0; index < 7; index += 1) {
    const angle = -2.2 + (index / 6) * 4.4;
    const tick = addBox(
      climate,
      [0.035, 0.13, 0.035],
      [Math.sin(angle) * 0.37, 0.58 + Math.cos(angle) * 0.37, 0.19],
      brass,
    );
    tick.rotation.z = -angle;
  }

  const airflow = [];
  for (let index = 0; index < 10; index += 1) {
    const airflowMaterial = material(0x65c8ef, {
      emissive: 0x65c8ef,
      emissiveIntensity: 0.9,
      transparent: true,
      opacity: 0.2,
      roughness: 0.25,
    });
    const particle = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), airflowMaterial);
    particle.userData.offset = index / 10;
    climate.add(particle);
    airflow.push(particle);
  }
  return { group: climate, dialFace, needlePivot, airflow };
}

function createChandelier() {
  const chandelier = new THREE.Group();
  const brass = material(0x8c6a32, { metalness: 0.62, roughness: 0.34 });
  addCylinder(chandelier, 0.055, 1.5, [0, 0.75, 0], brass, 12);
  addCylinder(chandelier, 0.13, 0.42, [0, 0.02, 0], brass, 12);
  const bulbMaterials = [];
  for (let index = 0; index < 6; index += 1) {
    const angle = (index / 6) * Math.PI * 2;
    const x = Math.cos(angle) * 1.05;
    const z = Math.sin(angle) * 1.05;
    const arm = addBox(chandelier, [1.1, 0.07, 0.07], [x / 2, -0.08, z / 2], brass, -angle);
    arm.lookAt(x, -0.08, z);
    const bulbMaterial = material(0xffdf9b, {
      emissive: 0xffdf9b,
      emissiveIntensity: 0.9,
      roughness: 0.28,
    });
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), bulbMaterial);
    bulb.position.set(x, -0.12, z);
    chandelier.add(bulb);
    bulbMaterials.push(bulbMaterial);
  }
  const light = new THREE.PointLight(0xffd6a0, 4.0, 15, 1.7);
  light.position.set(0, -0.35, 0);
  chandelier.add(light);
  return { group: chandelier, light, bulbMaterials };
}

function createSconce(position) {
  const group = new THREE.Group();
  group.position.set(...position);
  const brass = material(0x8c6a32, { metalness: 0.6, roughness: 0.34 });
  addBox(group, [0.18, 0.62, 0.12], [0, 0, 0], brass);
  const bulbMaterial = material(0xffdf9b, {
    emissive: 0xffdf9b,
    emissiveIntensity: 0.8,
    roughness: 0.3,
  });
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), bulbMaterial);
  bulb.position.set(0, 0.38, 0.18);
  group.add(bulb);
  const light = new THREE.PointLight(0xffd6a0, 1.5, 7, 1.8);
  light.position.set(0, 0.35, 0.55);
  group.add(light);
  return { group, light, bulbMaterial };
}

export function createWayneManorScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080a11);
  scene.fog = new THREE.FogExp2(0x080a11, 0.013);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 70);
  camera.position.set(11.5, 14.5, 13.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  container.append(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 16;
  controls.maxDistance = 27;
  controls.minPolarAngle = 0.5;
  controls.maxPolarAngle = 1.08;
  controls.target.set(0, 0.3, 0);

  // Keep the fireplace-led nighttime look, with a touch more fill so room details remain visible.
  scene.add(new THREE.AmbientLight(0x21191a, 0.68));
  scene.add(new THREE.HemisphereLight(0x1b2230, 0x1c0f09, 0.42));
  const electricFill = new THREE.HemisphereLight(0xffd6a0, 0x3b241a, 0.65);
  scene.add(electricFill);
  scene.add(createManorRoom());

  const { group: fireplace, fireLight, flames } = createFireplace();
  fireplace.position.set(0, 0, -4.55);
  scene.add(fireplace);

  const portrait = createPortrait();
  portrait.position.set(0, 4.04, -4.69);
  scene.add(portrait);

  const rug = new THREE.Mesh(
    new THREE.PlaneGeometry(7.5, 4.8),
    material(0x49212a, { roughness: 1, side: THREE.DoubleSide }),
  );
  rug.rotation.x = -Math.PI / 2;
  rug.position.set(0, 0.005, 0.75);
  scene.add(rug);
  const rugInset = new THREE.Mesh(
    new THREE.PlaneGeometry(6.8, 4.1),
    material(0x6c4d32, { roughness: 1, side: THREE.DoubleSide }),
  );
  rugInset.rotation.x = -Math.PI / 2;
  rugInset.position.set(0, 0.012, 0.75);
  scene.add(rugInset);

  const sofa = createSofa();
  sofa.position.set(0, 0, 2.55);
  scene.add(sofa);

  const leftChair = createChair();
  leftChair.position.set(-4.25, 0, 0.35);
  leftChair.rotation.y = -Math.PI / 2;
  scene.add(leftChair);
  const rightChair = createChair();
  rightChair.position.set(4.25, 0, 0.35);
  rightChair.rotation.y = Math.PI / 2;
  scene.add(rightChair);

  const coffeeTable = createCoffeeTable();
  coffeeTable.position.set(0, 0, 0.45);
  scene.add(coffeeTable);

  const telephoneTable = createTelephoneTable();
  telephoneTable.position.set(3.45, 0, 2.55);
  scene.add(telephoneTable);

  const telephone = createTelephone();
  telephone.group.position.set(3.45, 0.82, 2.55);
  telephone.group.rotation.y = -0.16;
  scene.add(telephone.group);

  const turntableTable = createTurntableTable();
  turntableTable.position.set(-3.45, 0, 2.55);
  scene.add(turntableTable);

  const turntable = createTurntable();
  turntable.group.position.set(-3.45, 0.84, 2.55);
  turntable.group.rotation.y = 0.16;
  scene.add(turntable.group);

  const leftBookcase = createBookcase();
  leftBookcase.position.set(-5.7, 0, -4.42);
  scene.add(leftBookcase);
  const rightBookcase = createBookcase();
  rightBookcase.position.set(5.7, 0, -4.42);
  scene.add(rightBookcase);

  const climateSystem = createClimateSystem();
  climateSystem.group.position.set(-6.78, 2.05, 1.4);
  climateSystem.group.rotation.y = Math.PI / 2;
  scene.add(climateSystem.group);

  const chandelier = createChandelier();
  chandelier.group.position.set(0, 4.4, 0.45);
  scene.add(chandelier.group);
  const sconces = [createSconce([-2.75, 2.75, -4.68]), createSconce([2.75, 2.75, -4.68])];
  for (const sconce of sconces) scene.add(sconce.group);

  const state = {
    intensity: 0.65,
    targetIntensity: 0.65,
    color: new THREE.Color(0xffd6a0),
    targetColor: new THREE.Color(0xffd6a0),
    temperature: 0.57,
    targetTemperature: 0.57,
    temperatureActivity: 0,
    telephoneRinging: false,
    mediaPlaying: false,
  };

  function resize() {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }

  function applyWorldState(world) {
    const light = world.lights[world.default_light_group];
    if (light) {
      state.targetIntensity = light.effective_intensity;
      state.targetColor.set(light.color_hex);
    }
    const thermostat = world.thermostats[world.default_thermostat];
    if (thermostat) {
      const span = thermostat.maximum_setpoint - thermostat.minimum_setpoint;
      const targetTemperature = THREE.MathUtils.clamp(
        (thermostat.setpoint - thermostat.minimum_setpoint) / span,
        0,
        1,
      );
      if (Math.abs(targetTemperature - state.targetTemperature) > 0.001) {
        state.temperatureActivity = 1;
      }
      state.targetTemperature = targetTemperature;
    }
    state.telephoneRinging = Boolean(world.telephone?.ringing);
    state.mediaPlaying = Boolean(
      world.media?.connected && world.media?.playback_state === "PLAYING",
    );
  }

  const cool = new THREE.Color(0x57b8df);
  const warm = new THREE.Color(0xf0a35a);
  const climateColor = new THREE.Color();
  const clock = new THREE.Clock();
  let frame = 0;
  let elapsed = 0;
  function animate() {
    frame = window.requestAnimationFrame(animate);
    const deltaSeconds = Math.min(clock.getDelta(), 0.1);
    const interpolation = 1 - Math.exp(-5 * deltaSeconds);
    elapsed += deltaSeconds;
    controls.update();
    state.intensity = THREE.MathUtils.lerp(state.intensity, state.targetIntensity, interpolation);
    state.color.lerp(state.targetColor, interpolation);
    state.temperature = THREE.MathUtils.lerp(
      state.temperature,
      state.targetTemperature,
      interpolation,
    );
    state.temperatureActivity *= Math.exp(-1.1 * deltaSeconds);

    chandelier.light.intensity = state.intensity * 32;
    chandelier.light.color.copy(state.color);
    electricFill.intensity = state.intensity * 1.45;
    electricFill.color.copy(state.color);
    for (const bulbMaterial of chandelier.bulbMaterials) {
      bulbMaterial.emissive.copy(state.color);
      bulbMaterial.emissiveIntensity = 0.05 + state.intensity * 2.1;
    }
    for (const sconce of sconces) {
      sconce.light.intensity = state.intensity * 15;
      sconce.light.color.copy(state.color);
      sconce.bulbMaterial.emissive.copy(state.color);
      sconce.bulbMaterial.emissiveIntensity = 0.04 + state.intensity * 1.8;
    }

    const flicker = 0.9 + Math.sin(elapsed * 7.1) * 0.055 + Math.sin(elapsed * 12.7) * 0.035;
    fireLight.intensity = 13.0 * flicker;
    for (let index = 0; index < flames.length; index += 1) {
      flames[index].scale.y = (1.8 + (index % 2) * 0.65) * (0.92 + 0.08 * Math.sin(elapsed * 8 + index));
    }

    climateColor.copy(cool).lerp(warm, state.temperature);
    climateSystem.dialFace.material.color.copy(climateColor);
    climateSystem.dialFace.material.emissive.copy(climateColor);
    climateSystem.needlePivot.rotation.z = THREE.MathUtils.lerp(2.2, -2.2, state.temperature);
    for (const particle of climateSystem.airflow) {
      const progress = (elapsed * 0.22 + particle.userData.offset) % 1;
      particle.position.set(
        Math.sin(progress * Math.PI * 2 + particle.userData.offset * 5) * 0.72,
        -1.22 + Math.sin(progress * Math.PI) * 0.2,
        0.35 + progress * 2.35,
      );
      particle.material.color.copy(climateColor);
      particle.material.emissive.copy(climateColor);
      particle.material.opacity =
        Math.sin(progress * Math.PI) * (0.08 + state.temperatureActivity * 0.62);
    }
    if (state.telephoneRinging) {
      telephone.group.rotation.z = Math.sin(elapsed * 34) * 0.035;
      telephone.handset.position.y = 0.46 + Math.abs(Math.sin(elapsed * 17)) * 0.025;
      telephone.indicator.emissiveIntensity = 0.8 + Math.abs(Math.sin(elapsed * 8)) * 1.4;
    } else {
      telephone.group.rotation.z = THREE.MathUtils.lerp(
        telephone.group.rotation.z,
        0,
        interpolation,
      );
      telephone.handset.position.y = THREE.MathUtils.lerp(
        telephone.handset.position.y,
        0.46,
        interpolation,
      );
      telephone.indicator.emissiveIntensity = 0.08;
    }
    if (state.mediaPlaying) {
      const rotation = deltaSeconds * Math.PI * 0.9;
      turntable.record.rotation.y += rotation;
      turntable.recordLabel.rotation.y += rotation;
      turntable.tonearm.rotation.y = THREE.MathUtils.lerp(
        turntable.tonearm.rotation.y,
        -0.18,
        interpolation,
      );
      turntable.indicator.emissiveIntensity = 1.1;
    } else {
      turntable.tonearm.rotation.y = THREE.MathUtils.lerp(
        turntable.tonearm.rotation.y,
        0.12,
        interpolation,
      );
      turntable.indicator.emissiveIntensity = 0.05;
    }
    renderer.render(scene, camera);
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();
  animate();

  return {
    applyWorldState,
    dispose() {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        object.geometry?.dispose();
        if (Array.isArray(object.material)) {
          for (const objectMaterial of object.material) objectMaterial.dispose();
        } else {
          object.material?.dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
