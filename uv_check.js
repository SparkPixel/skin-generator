const THREE = require('three');
const geo = new THREE.BoxGeometry(1, 1, 1);
const pos = geo.attributes.position;
function printFace(name, f) {
  console.log("Face " + name + " (index " + f + "):");
  for (let i = 0; i < 4; i++) {
    const idx = f * 4 + i;
    console.log(`  V${i}: [${pos.getX(idx)}, ${pos.getY(idx)}, ${pos.getZ(idx)}]`);
  }
}
printFace("+X (face 0)", 0);
printFace("-X (face 1)", 1);
printFace("+Y (face 2, TOP)", 2);
printFace("-Y (face 3, BOTTOM)", 3);
printFace("+Z (face 4, FRONT)", 4);
printFace("-Z (face 5, BACK)", 5);
