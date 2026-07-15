#ifndef PPU_H
#define PPU_H

#include "../types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "saveload.h"

typedef struct Ppu Ppu;

typedef struct BgLayer {
  uint16_t xhScroll;
  uint16_t xvScroll;
  bool xtilemapWider;
  bool xtilemapHigher;
  uint16_t xtilemapAdr;
  uint16_t xtileAdr;
  bool xxbigTiles;
  bool xxmosaicEnabled;
} BgLayer;

enum {
  kPpuXPixels = 256,
  // Maximum widescreen expansion *per side*, baked into the priority-buffer
  // capacity. This is a compile-time ceiling only; the actual extra columns
  // rendered each frame are the runtime ppu->extraLeftCur/extraRightCur, which
  // default to 0 (authentic 256-wide output). 96 per side allows up to a
  // 448-pixel internal width, comfortably past 16:9 at 224 lines.
  kPpuExtraLeftRight = 96,
  // Full internal width of the priority buffers (logical 256 + both borders).
  kPpuBufWidth = kPpuXPixels + kPpuExtraLeftRight * 2,
};

typedef uint16_t PpuZbufType;

typedef struct PpuPixelPrioBufs {
  // This holds the prio in the upper 8 bits and the color in the lower 8 bits.
  // Sized for the widescreen border; logical screen x maps to
  // data[x + kPpuExtraLeftRight].
  PpuZbufType data[kPpuBufWidth];
} PpuPixelPrioBufs;

enum {
  kPpuRenderFlags_NewRenderer = 1,
  // Render mode7 upsampled by 4x4
  kPpuRenderFlags_4x4Mode7 = 2,
  // Use 240 height instead of 224
  kPpuRenderFlags_Height240 = 4,
  // Disable sprite render limits
  kPpuRenderFlags_NoSpriteLimits = 8,
};

typedef struct Layer {
  bool xmainScreenEnabled;
  bool xsubScreenEnabled;
  bool xmainScreenWindowed;
  bool xsubScreenWindowed;
} Layer;

typedef struct WindowLayer {
  bool xwindow1enabled;
  bool xwindow2enabled;
  bool xwindow1inversed;
  bool xwindow2inversed;
  uint8_t xmaskLogic;
} WindowLayer;

#define PPU_SAVESTATE_REGS_SIZE 0x40
#define PPU_SAVESTATE_MEM_SIZE 0x10420

struct Ppu {
  // Snes registers. Saved to snapshot. Need to be stable
  // -- START OF SNAPSHOT, 0x40 bytes
  uint8 inidisp;
  uint8 obsel;
  uint8 oamaddl;
  uint8 oamaddh;
  uint8 bgmode;
  uint8 mosaic;
  uint8 bgXsc[4];
  uint16 bgTileAdr;
  uint8 m7sel;
  uint8 setini;
  uint16 hScroll[4];
  uint16 vScroll[4];
  int16_t m7matrix[8]; // a, b, c, d, x, y, h, v
  uint16 fixedColor;
  uint32 windowsel;
  uint8 window1left;
  uint8 window1right;
  uint8 window2left;
  uint8 window2right;
  uint16 wbgobjlog;
  uint8 screenEnabled[2];
  uint8 screenWindowed[2];
  uint8 cgadsub;
  uint8 cgwsel;
  // -- END OF SNAPSHOT

  // vram access
  uint16_t vramPointer;
  bool vramIncrementOnHigh;
  uint8_t vramRemapMode;
  uint8_t vramIncrement;
  uint16_t vramReadBuffer;
  // cgram access
  uint8_t cgramPointer;
  bool cgramSecondWrite;
  uint8_t cgramBuffer;
  // oam access
  uint8_t oamAdr;
  bool oamInHigh;
  bool oamSecondWrite;
  uint8_t oamBuffer;
  bool timeOver;
  bool rangeOver;
  uint8_t scrollPrev;
  uint8_t scrollPrev2;
  uint8_t mosaicStartLine;
  uint8_t m7prev;
  // mode 7 internal
  int32_t m7startX;
  int32_t m7startY;
  // settings
  bool evenFrame;
  bool frameOverscan; // if we are overscanning this frame (determined at 0,225)
  bool frameInterlace; // if we are interlacing this frame (determined at start vblank)
  // latching
  uint16_t hCount;
  uint16_t vCount;
  bool hCountSecond;
  bool vCountSecond;
  bool countersLatched;
  // pixel buffer (xbgr)
  // times 2 for even and odd frame

  uint8_t extraLeftCur, extraRightCur, extraLeftRight, extraBottomCur;
  // Widescreen HUD split (see PpuSetWidescreenHudSplit). 0 height = off.
  uint8_t wsHudSplitHeight, wsHudLeftEnd, wsHudRightStart;
  // Widescreen BG3 widen (see PpuSetWidescreenBg3Widen). Scanlines >= this let
  // BG3 (layer 2) extend into the side margins like BG1/BG2 instead of staying
  // clamped to the authentic 256-wide region. 0 = off (BG3 clamped everywhere,
  // so a BG3 status bar never tiles into the margins). SMW sets it to the HUD
  // band height so water/level content on BG3 below the bar fills 16:9.
  uint8_t wsBg3WidenY;
  // Widescreen per-layer clamp (see PpuSetWidescreenLayerClamp). Bit L set =>
  // BGL+1 (layer L, 0..3) is clamped to the authentic 256-wide region even in
  // widescreen. For UI/dialog/status layers whose tilemap is only 256 wide, so
  // they never tile wrapped/garbage columns into the margins while the world
  // layers beside them stay wide. 0 = every layer extended (default).
  uint8_t wsLayerClamp;
  // Widescreen per-layer mirror fill (see PpuSetWidescreenLayerMirror). Bit L
  // keeps BGL+1 authentic in the center, then reflects its rendered edge pixels
  // into the side margins. Used for decorative 256-wide layers that have no
  // real offscreen world data. 0 = disabled.
  uint8_t wsLayerMirror;
  // Widescreen per-layer repeat fill (see PpuSetWidescreenLayerRepeat). Bit L
  // keeps BGL+1 authentic in the center, then cyclically continues that
  // rendered 256px scanline into the margins. Unlike reflection, this keeps
  // raster/HDMA parallax moving in the same direction across the seam.
  uint8_t wsLayerRepeat;
  // Per-layer widescreen clamp BAND (see PpuSetWidescreenLayerClampBand): on
  // scanlines [y0,y1) layer L is clamped to the authentic 256, while it still
  // extends into the margins outside that band. This is the generic "overlay
  // plane" ("BG2.5"): a bounded UI element (dialog box, menu panel) sharing a
  // layer with genuinely-wide world content (pillars) — clamp only the rows the
  // UI occupies so the world layer stays wide above/below it. y1<=y0 = off.
  uint8_t wsClampY0[4], wsClampY1[4];
  // Per-layer cyclic-repeat BAND (see PpuSetWidescreenLayerRepeatBand): on
  // scanlines [y0,y1), repeat the authentic rendered scanline into both
  // margins. This takes precedence over a whole-layer clamp on those rows,
  // allowing animated/raster content to share a BG with bounded scenery.
  uint8_t wsRepeatY0[4], wsRepeatY1[4];
  // Widescreen margin source gap per layer, in pixels per side (see
  // PpuSetWidescreenLayerMarginGap): margins skip the first N offscreen
  // columns (games' UI staging area) and sample the tilemap beyond them.
  uint8_t wsMarginGapL[4], wsMarginGapR[4];
  uint8_t lastMosaicModulo;
  uint8_t lastBrightnessMult;
  bool lineHasSprites;
  // kPpuRenderFlags_* for this session (PpuBeginDrawing). NoSpriteLimits
  // lifts the hardware 32-sprites/34-tiles per-scanline caps — on widescreen
  // lines with more sprites visible, the authentic caps clip sprites EARLIER
  // than a real console would relative to the wider view.
  uint32_t renderFlags;
  PpuPixelPrioBufs bgBuffers[2];
  PpuPixelPrioBufs objBuffer;
  uint32_t renderPitch;
  uint8_t *renderBuffer;
  uint8_t brightnessMult[32 + 31];
  uint8_t brightnessMultHalf[32 * 2];
  uint8_t mosaicModulo[kPpuXPixels];

  void *pad2;

  // -- START OF SNAPSHOT, 0x10420 bytes
  uint16_t cgram[0x100];
  uint16_t oam[0x100];
  uint8_t highOam[0x20];
  uint16_t vram[0x8000];
  // -- END OF SNAPSHOT


};

#define SPRITE_PRIO_TO_PRIO(prio, level6) (((prio) * 4 + 2) * 16 + 4 + (level6 ? 2 : 0))
#define SPRITE_PRIO_TO_PRIO_HI(prio) ((prio) * 4 + 2)

#define IS_SCREEN_ENABLED(ppu, sub, layer) (ppu->screenEnabled[sub] & (1 << layer))
#define IS_SCREEN_WINDOWED(ppu, sub, layer) (ppu->screenWindowed[sub] & (1 << layer))
#define GET_WINDOW_FLAGS(ppu, layer) (ppu->windowsel >> (layer * 4))

#define PPU_brightness(ppu) (ppu->inidisp & 0xf)
#define PPU_forcedBlank(ppu) (ppu->inidisp & 0x80)

#define PPU_objSize(ppu) (ppu->obsel >> 5)
#define PPU_objTileAdr1(ppu) ((ppu->obsel & 7) << 13)
#define PPU_objTileAdr2(ppu) (PPU_objTileAdr1(ppu) + (((ppu->obsel & 0x18) + 8) << 9))

#define PPU_objPriority(ppu) (ppu->oamaddh & 0x80)

#define PPU_mode(ppu) (ppu->bgmode & 7)
#define PPU_bg3priority(ppu) (ppu->bgmode & 0x8)
#define PPU_bigTiles(ppu, layer) (ppu->bgmode >> layer & 0x10)

#define PPU_mosaicEnabled(ppu, layer) (ppu->mosaic & (1 << layer))
#define PPU_mosaicSize(ppu) ((ppu->mosaic >> 4) + 1)

#define PPU_bgTilemapWider(ppu, layer) (ppu->bgXsc[layer] & 0x1)
#define PPU_bgTilemapHigher(ppu, layer) (ppu->bgXsc[layer] & 0x2)
#define PPU_bgTilemapAdr(ppu, layer) ((ppu->bgXsc[layer] & 0xfc) << 8)
#define PPU_bgTileAdr(ppu, layer) ((ppu->bgTileAdr >> (layer * 4) & 0xf) << 12)

#define PPU_m7xFlip(ppu) (ppu->m7sel & 0x1)
#define PPU_m7yFlip(ppu) (ppu->m7sel & 0x2)
#define PPU_m7charFill(ppu) (ppu->m7sel & 0x40)
#define PPU_m7largeField(ppu) (ppu->m7sel & 0x80)

#define PPU_directColor(ppu) ((ppu->cgwsel & 0x1) != 0)
#define PPU_addSubscreen(ppu) ((ppu->cgwsel & 0x2) != 0)
#define PPU_preventMathMode(ppu) (ppu->cgwsel >> 4 & 0x3)
#define PPU_clipMode(ppu) (ppu->cgwsel >> 6 & 0x3)

#define PPU_mathEnabled(ppu) (ppu->cgadsub & 0x3f)
#define PPU_halfColor(ppu) ((ppu->cgadsub & 0x40) != 0)
#define PPU_subtractColor(ppu) ((ppu->cgadsub & 0x80) != 0)

#define PPU_fixedColorR(ppu) (ppu->fixedColor & 0x1f)
#define PPU_fixedColorG(ppu) (ppu->fixedColor >> 5 & 0x1f)
#define PPU_fixedColorB(ppu) (ppu->fixedColor >> 10 & 0x1f)

#define PPU_interlace(ppu) ((ppu->setini & 0x1) != 0)
#define PPU_objInterlace(ppu) ((ppu->setini & 0x2) != 0)
#define PPU_overscan(ppu) ((ppu->setini & 0x4) != 0)
#define PPU_pseudoHires(ppu) ((ppu->setini & 0x8) != 0)
#define PPU_m7extBg(ppu) ((ppu->setini & 0x40) != 0)


enum {
  kWindow1Inversed = 1,
  kWindow1Enabled = 2,
  kWindow2Inversed = 4,
  kWindow2Enabled = 8,
};


Ppu* ppu_init(void);
void ppu_free(Ppu* ppu);
void ppu_reset(Ppu* ppu);
bool ppu_checkOverscan(Ppu* ppu);
void ppu_handleVblank(Ppu* ppu);
void ppu_runLine(Ppu* ppu, int line);
uint8_t ppu_read(Ppu* ppu, uint8_t adr);
void ppu_write(Ppu* ppu, uint8_t adr, uint8_t val);
void ppu_saveload(Ppu *ppu, SaveLoadInfo *sli);
void PpuBeginDrawing(Ppu *ppu, uint8_t *pixels, size_t pitch, uint32_t render_flags);

// Set the symmetric widescreen border, in pixels per side (clamped to
// kPpuExtraLeftRight). 0 restores authentic 256-wide rendering. The internal
// render width becomes 256 + 2*extra. Drives the dormant extraLeftCur/
// extraRightCur/extraLeftRight machinery used by the line renderer.
void PpuSetExtraSpace(Ppu *ppu, uint8_t extra);

// Render authentic 256-wide content centered within a `budget`-per-side wider
// framebuffer (no border columns drawn). For bounded screens; caller blacks
// out the side margins to pillarbox.
void PpuSetExtraSpaceCentered(Ppu *ppu, uint8_t budget);

// Asymmetric per-side widescreen margin (the snesrev/zelda3 model, see
// attribution in IMPROVEMENTS.md). The centering budget (extraLeftRight) must
// already be set via PpuSetExtraSpaceCentered/PpuSetExtraSpace; this fills the
// per-frame extraLeftCur/extraRightCur/extraBottomCur within that budget,
// clamped so the window/sprite/composite paths never read past the
// priority-buffer capacity (left/right) or the 16px overscan bottom. Negative
// inputs clamp to 0. (0,0,0) collapses to a centered pillarbox. Callers
// re-apply per frame (ppu_reset zeroes the fields). Used by games whose own
// scroll/room-bounds state drives the visible margin dynamically (Zelda),
// versus PpuSetExtraSpace's fixed symmetric border (SMW).
void PpuSetExtraSideSpace(Ppu *ppu, int left, int right, int bottom);

// Widescreen HUD split (opt-in, configured by the game frontend): for
// scanlines < height, BG3 (layer 2) is drawn as three chunks — source
// [0,left_end) anchored to the LEFT border edge, [left_end,right_start)
// kept centered (unmoved), [right_start,256) anchored to the RIGHT border
// edge. The vacated spans stay transparent. height 0 = off (authentic).
// Only takes effect while extra border columns are active and BG3 is not
// windowed; mosaic lines fall back to centered. Like the extra-space
// setters, callers re-apply per frame (ppu_reset zeroes the fields).
void PpuSetWidescreenHudSplit(Ppu *ppu, uint8_t height, uint8_t left_end,
                              uint8_t right_start);

// Let BG3 (layer 2) render into the widescreen side margins on scanlines
// >= from_y, instead of being clamped to the authentic 256-wide region. Pass
// the HUD band height so the status bar above it stays clamped (or split) while
// level content on BG3 below it (e.g. SMW water) fills 16:9. from_y 0 = off.
// Like the other widescreen setters, callers re-apply per frame.
void PpuSetWidescreenBg3Widen(Ppu *ppu, uint8_t from_y);

// Per-layer widescreen clamp: bit L (0..3) keeps BG(L+1) in the authentic 256
// columns even while other layers extend into the margins. For scenes that mix
// genuinely-wide world layers with 256-wide UI/dialog/status layers (or layers
// whose offscreen tilemap data is not meant to be shown) — clamp the latter so
// they never tile wrapped/garbage columns into the border. mask 0 = every layer
// extended (default). Independent of the BG3-specific widen/split controls; a
// layer clamped here is clamped regardless of wsBg3WidenY. Re-apply per frame.
void PpuSetWidescreenLayerClamp(Ppu *ppu, uint8_t mask);

// Mirror-fill BG-layer side margins from the authentic 256-wide rendered
// result. Bit L reflects BG(L+1) without duplicating the boundary pixel:
// left destination x<0 samples -x, right destination x>=256 samples 510-x.
// Reflection happens after tile decode/windowing but before layer priority and
// color math are finalized, so transparency, priority, palette animation, and
// sub-screen behavior remain layer-correct. The current implementation applies
// to Mode-1 4bpp BG1/BG2; unsupported layers remain authentically clamped.
// Re-apply per frame. A mirror bit takes visual precedence over a clamp bit.
void PpuSetWidescreenLayerMirror(Ppu *ppu, uint8_t mask);

// Repeat-fill BG-layer side margins from the authentic 256-wide rendered
// result. Left x<0 samples 256+x; right x>=256 samples x-256. Because this is
// performed independently on each already-rendered scanline, per-line HDMA
// scroll, transparency, priority, palette animation, and color math remain
// layer-correct. The current implementation applies to Mode-1 4bpp BG1/BG2;
// unsupported layers remain authentically clamped. Re-apply per frame. A
// repeat bit takes precedence if the same layer is also marked for mirroring.
void PpuSetWidescreenLayerRepeat(Ppu *ppu, uint8_t mask);

// Clamp BG(layer+1) to the authentic 256 on scanlines [y0,y1) only (the generic
// "overlay plane" / BG2.5): a bounded UI element sharing a layer with wide world
// content is confined to the center on its own rows, while the layer stays wide
// above and below. y1<=y0 disables. Re-apply per frame (the extra-space setters
// reset it). Independent of the whole-layer clamp and the BG3 widen/split.
void PpuSetWidescreenLayerClampBand(Ppu *ppu, uint8_t layer, uint8_t y0,
                                    uint8_t y1);

// Cyclically repeat BG(layer+1)'s authentic rendered scanline into the margins
// on scanlines [y0,y1) only. This is the banded form of
// PpuSetWidescreenLayerRepeat: it preserves per-line scroll, tile animation,
// transparency, priority, and color math, and takes precedence over a
// whole-layer clamp for the selected rows. Currently supported by the Mode-1
// 4bpp BG1/BG2 path. y1<=y0 disables. Re-apply per frame.
void PpuSetWidescreenLayerRepeatBand(Ppu *ppu, uint8_t layer, uint8_t y0,
                                     uint8_t y1);

// Widescreen margin source gap: the margins of BG(layer+1) skip the first
// left_px/right_px pixels past the authentic screen edges and sample the
// tilemap beyond them. For games that park UI-construction tiles in the
// offscreen columns adjacent to the visible screen (a staging area that
// hardware never shows but widescreen margins would expose) — the gap keeps
// the staging strip invisible while the layer still fills the margins with
// its real repeating/world content. Applies to the 4bpp/2bpp BG paths
// (mosaic lines fall back to ungapped). 0/0 = off. Re-apply per frame.
void PpuSetWidescreenLayerMarginGap(Ppu *ppu, uint8_t layer, uint8_t left_px,
                                    uint8_t right_px);

int PpuGetCurrentRenderScale(Ppu *ppu, uint32_t render_flags);

#endif
