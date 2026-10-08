// World coordinates: east and north in meters; headings clockwise from north.
export const BLOCK = 160;
export const DIRECTIONS = ['Northbound', 'Eastbound', 'Southbound', 'Westbound'];
export const INITIAL_FRAME = Object.freeze({east: 0, north: 0, heading: 0, offset: 100});
export const mod = (x, n) => ((x % n) + n) % n;
export function basis(heading) {
  const h = heading * Math.PI / 2;
  return {fe: Math.round(Math.sin(h)), fn: Math.round(Math.cos(h)), re: Math.round(Math.cos(h)), rn: -Math.round(Math.sin(h))};
}
export function worldPoint(frame, x, s) {
  const b = basis(frame.heading);
  return {east: frame.east + b.fe * s + b.re * x, north: frame.north + b.fn * s + b.rn * x};
}
export function localPoint(frame, point) {
  const b = basis(frame.heading), de = point.east - frame.east, dn = point.north - frame.north;
  return {x: de * b.re + dn * b.rn, s: de * b.fe + dn * b.fn};
}
export function junctionKey(point) { return `${Math.round(point.east / BLOCK)},${Math.round((point.north - 100) / BLOCK)}`; }
export function streetName(frame) {
  const avenues = ['Linden Avenue', 'Cedar Avenue', 'Willow Avenue', 'Ash Avenue', 'Birch Avenue'];
  const streets = ['Market Street', 'Harbor Street', 'Garden Street', 'Station Street', 'Park Street'];
  return frame.heading % 2 === 0 ? avenues[mod(Math.round(frame.east / BLOCK), avenues.length)]
    : streets[mod(Math.round((frame.north - 100) / BLOCK), streets.length)];
}
export function nextCenter(s, offset = 100) { return offset + Math.ceil((s - offset - 16) / BLOCK) * BLOCK; }
export function junctionSignal(point, heading, time, forced = null) {
  const i = Math.round(point.east / BLOCK), j = Math.round((point.north - 100) / BLOCK);
  const phase = mod(time + i * 5 + j * 7, 36);
  if (forced && forced.key === junctionKey(point) && time < forced.until) {
    return {color: 'red', remaining: forced.until - time};
  }
  if (heading % 2 === 0) return phase < 19 ? {color: 'green', remaining: 19 - phase}
    : phase < 22 ? {color: 'amber', remaining: 22 - phase} : {color: 'red', remaining: 36 - phase};
  return phase < 24 ? {color: 'red', remaining: 24 - phase}
    : phase < 32 ? {color: 'green', remaining: 32 - phase}
    : phase < 34 ? {color: 'amber', remaining: 34 - phase} : {color: 'red', remaining: 60 - phase};
}
export function outgoingFrame(frame, center, direction) {
  const p = worldPoint(frame, 0, center);
  return {...p, heading: mod(frame.heading + (direction === 'left' ? -1 : direction === 'right' ? 1 : 0), 4), offset: 0};
}
// Shortest route on the street grid, including heading: U-turns are unavailable.
// The first point is the next junction, so an in-progress maneuver stays committed.
export function routeToDestination(start, destination) {
  const origin = {i: Math.round(start.east / BLOCK), j: Math.round((start.north - 100) / BLOCK), heading: start.heading};
  const goal = {i: Math.round(destination.east / BLOCK), j: Math.round((destination.north - 100) / BLOCK)};
  const key = n => `${n.i},${n.j},${n.heading}`;
  const minI = Math.min(origin.i, goal.i) - 2, maxI = Math.max(origin.i, goal.i) + 2;
  const minJ = Math.min(origin.j, goal.j) - 2, maxJ = Math.max(origin.j, goal.j) + 2;
  const queue = [{...origin, parent: -1}], seen = new Set([key(origin)]);
  for (let index = 0; index < queue.length; index++) {
    const node = queue[index];
    if (node.i === goal.i && node.j === goal.j) {
      const route = [];
      for (let at = index; at >= 0; at = queue[at].parent) {
        const n = queue[at];
        route.push({east: n.i * BLOCK, north: 100 + n.j * BLOCK, direction: null});
        if (n.parent >= 0) route[route.length - 1].incoming = n.direction;
      }
      route.reverse();
      for (let i = 0; i < route.length - 1; i++) route[i].direction = route[i + 1].incoming;
      return route;
    }
    for (const direction of ['straight', 'left', 'right']) {
      const heading = mod(node.heading + (direction === 'left' ? -1 : direction === 'right' ? 1 : 0), 4);
      const b = basis(heading), next = {i: node.i + b.fe, j: node.j + b.fn, heading, parent: index, direction};
      if (next.i < minI || next.i > maxI || next.j < minJ || next.j > maxJ || seen.has(key(next))) continue;
      seen.add(key(next));queue.push(next);
    }
  }
  return [];
}
export function turnGeometry(frame, center, direction) {
  const radius = direction === 'right' ? 8 : 16;
  return {frame: {...frame}, center, direction, radius, length: radius * Math.PI / 2, progress: 0,
    exitFrame: outgoingFrame(frame, center, direction), lane: direction === 'right' ? 1 : 0};
}
export function turnPose(turn, progress = turn.progress) {
  const angle = Math.min(Math.PI / 2, Math.max(0, progress / turn.radius));
  const right = turn.direction === 'right';
  const x = right ? 14 - turn.radius * Math.cos(angle) : -14 + turn.radius * Math.cos(angle);
  const s = turn.center - 14 + turn.radius * Math.sin(angle);
  return {...worldPoint(turn.frame, x, s), heading: turn.frame.heading * Math.PI / 2 + (right ? angle : -angle)};
}
