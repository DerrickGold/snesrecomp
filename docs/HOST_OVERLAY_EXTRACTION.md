# Host-overlay extraction

The new PPU can export selected, already-rendered SNES graphics into transparent
ARGB surfaces without modifying emulated VRAM, OAM, WRAM, registers, DMA, or
savestate data. This is a renderer capability; each game still owns the policy
that identifies a HUD, portrait, logo, menu panel, or other promotable graphic.

## Boundary of responsibility

| Owner | Responsibility |
|---|---|
| PPU runner | Isolate BG1-BG4/OBJ pixels, apply tile decode, scroll, windows, mosaic, palette and master brightness, optionally omit the captured rectangle from the game framebuffer |
| Game policy | Decide the source, screen-space rectangle, OAM slots, and game states in which capture is valid |
| Host frontend | Crop pieces from the surface, anchor/scale them, substitute higher-resolution art, and choose final presentation order |

The runner deliberately does not depend on SDL, OpenGL, or a particular host
layout. Independent post-upscale composition remains a frontend operation.

## Lifecycle

Bindings are persistent host resources; capture rectangles are per-frame game
policy:

```c
PpuClearOverlayBindings(ppu);
PpuBindOverlaySurface(ppu, kPpuOverlaySource_Bg3,
                      bg3_argb, framebuffer_pitch);
PpuBindOverlaySurface(ppu, kPpuOverlaySource_Obj,
                      obj_argb, framebuffer_pitch);

/* Once per emulated frame, before scanout. */
PpuClearOverlayCaptures(ppu);
PpuSetOverlayCapture(ppu, kPpuOverlaySource_Bg3,
                     0, 0, 256, 40,
                     kPpuOverlayFlag_RemoveFromGame);
PpuSetOverlayCapture(ppu, kPpuOverlaySource_Obj,
                     0, 0, 256, 40,
                     kPpuOverlayFlag_RemoveFromGame);
PpuSetOverlayOamRange(ppu, 0, 4);
```

Each source owns one bounding rectangle and one full-frame output surface. A
frontend may crop several disjoint graphics from that rectangle. Separate
sources can be captured simultaneously. Passing flags `0` makes a diagnostic
copy while retaining the source in the normal framebuffer;
`RemoveFromGame` promotes it by omitting the rectangle from both the SNES main
and subscreen paths.

Coordinates are visible screen space after scroll/window/mosaic processing.
The authentic screen is `x=[0,256)`; widescreen margins may use negative X or X
greater than 255. Output surfaces share the game framebuffer's pitch and
coordinate system, so authentic X zero is stored after the surface's left
centering budget.

OBJ capture additionally requires an OAM range. The runner treats slot identity
as opaque; the game must validate that the slots still represent the intended
graphic before selecting them. Rectangle clipping is per pixel, so partially
intersecting sprites retain their non-captured pixels in the normal OBJ plane.

## Rendering semantics

- Transparent BG/OBJ pixels remain alpha zero. Palette and master brightness
  are resolved before export.
- BG capture uses an isolated priority buffer. Outside a promoted rectangle,
  the isolated layer is merged back using the normal per-pixel priority word.
- The exported ARGB comes from the main-screen layer pass. The subscreen pass
  is isolated too when removal is requested, but is not exported; promoting a
  layer used only for subscreen color math needs a future screen-selection or
  intermediate-composition extension.
- Promotion is applied to main and subscreen so removed graphics cannot leave a
  color-math ghost underneath the host copy.
- Bindings survive `ppu_reset`; capture policy does not. A NULL binding is a
  deterministic no-op and preserves pure-headless/oracle output.
- The host overlay is above the already-flattened framebuffer. HUDs and topmost
  UI are therefore direct. Replacing scenery that must remain behind sprites or
  foreground BG priority requires exporting those occluding planes too, or a
  future intermediate-composition hook.

## Current coverage

The descriptor namespace includes BG1-BG4 and OBJ. The current new renderer
draws Mode 1 BG1-BG3 and Mode 7 BG1; those paths support capture now. BG4 becomes
active automatically when the renderer gains the SNES modes that actually draw
it. The old PPU renderer does not implement host-overlay extraction.

ActRaiser is the first consumer: it captures the authentic BG3 status rectangle
and a validated four-slot OBJ graphic, then performs its game-specific
left/center/right composition after SDL has upscaled the world. The resulting
host image is pixel-identical to the earlier HUD-specific implementation at
both Match Game and native-output 1x scales.

## Mode-7 canvas-space texture override

`PpuBindMode7OverlaySurface(ppu, pixels, pitch, scale)` binds a persistent
transparent ARGB surface covering the render frame at `scale` (1-4)
subsamples per axis. `PpuSetMode7Override(ppu, rgba, w, h, canvas rect)` is
per-frame game policy (cleared with the captures): main-screen Mode-7 BG1
pixels whose canvas coordinates fall inside the rectangle sample the given
texture instead of VRAM tiles — through the live matrix, so rotation, zoom,
per-scanline HDMA reloads, windows, and field wrap apply to the substituted
art. Each screen pixel emits scale x scale texture subsamples stepped at
fractional matrix increments (the same per-line register state, which is
exactly correct under HDMA). Opaque base samples (alpha >= 0x80) are removed
from both main and subscreen; translucent fringes stay in the surface for
the host to blend, and INIDISP master brightness is resolved on the samples.
The mosaic path renders authentically. ActRaiser's title intro swirl is the
first consumer.
