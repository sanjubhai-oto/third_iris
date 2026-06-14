#!/usr/bin/env python3
"""Generate a colosseum/arena GZ world (ground + sky + 2 tiers of pillars in a ring) with the chaser
(jetray) and target (jetray_target) above the centre. Run: python sim/gz/gen_colosseum.py"""
import math, os

HERE = os.path.dirname(os.path.abspath(__file__))

def ring(n, radius, h, z0, w, d, tag, color):
    s = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = radius * math.cos(a), radius * math.sin(a)
        s.append(f"""
    <model name="{tag}_{i}"><static>true</static><pose>{x:.2f} {y:.2f} {z0+h/2:.2f} 0 0 {a:.3f}</pose>
      <link name="l"><collision name="c"><geometry><box><size>{w} {d} {h}</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{w} {d} {h}</size></box></geometry>
        <material><diffuse>{color}</diffuse><specular>0.2 0.2 0.2 1</specular></material></visual></link></model>""")
    return "".join(s)

pillars = ring(24, 26, 10, 0, 3.2, 2.0, "outer", "0.78 0.72 0.58 1")      # outer colosseum wall
pillars += ring(20, 19, 7, 0, 2.4, 1.6, "inner", "0.70 0.64 0.50 1")      # inner tier
pillars += ring(16, 12, 4.5, 0, 1.6, 1.2, "arena", "0.66 0.60 0.46 1")    # arena ring

WORLD = f"""<?xml version="1.0" ?>
<!-- Colosseum/arena world for GZ up-cam air-to-air viewing: tiered rings of pillars around a central
     arena; chaser + target fly above the centre. Gives the 3D view real scenery and the up-cam parallax. -->
<sdf version="1.9">
  <world name="uav_colosseum">
    <physics type="ode"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.7 0.8 0.95 1</background><sky/></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 100 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse><specular>0.3 0.3 0.3 1</specular><direction>-0.3 0.2 -0.9</direction></light>
    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
        <material><diffuse>0.45 0.42 0.35 1</diffuse></material></visual></link></model>
    {pillars}
    <!-- SPECTATOR camera: 3rd-person view of the arena + drones, streamed to the browser (/spectator) -->
    <model name="spectator"><static>true</static><pose>34 0 16 0 -0.18 3.14159</pose>
      <link name="l"><sensor name="spec" type="camera"><topic>spectator</topic>
        <camera><horizontal_fov>1.5</horizontal_fov><image><width>1280</width><height>720</height></image>
          <clip><near>0.1</near><far>800</far></clip></camera>
        <always_on>1</always_on><update_rate>20</update_rate></sensor></link></model>
    <!-- camera-less visual drones (jetray has an onboard camera that competes with the spectator sensor
         on software GL); the spectator is the only render camera here -->
    <include><uri>model://jetray_target_blue</uri><name>chaser</name><pose>0 0 4 0 0 0</pose></include>
    <include><uri>model://jetray_target_red</uri><name>target</name><pose>0 0 18 0 0 0</pose></include>
  </world>
</sdf>
"""

out = os.path.join(HERE, "uav_colosseum.sdf")
open(out, "w").write(WORLD)
print("wrote", out, "with", WORLD.count("<model"), "models")
