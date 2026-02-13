# Frontend Modernization Master Guide — MIND Application

> **Audience**: Coding agents and developers.
> **Scope**: Master index for the frontend modernization project.

---

## 1. Documentation Map

| Artifact | Purpose |
|---|---|
| **[Design System](frontend_design_system.md)** | Defines the "Deep Cipher" color palette, Typography (Inter), and Icons (Phosphor). |
| **[Phase 1: Foundation](phase_1_foundation.md)** | ✅ `base.html` setup, CDN swap, CSS variables, Navbar modernization. |
| **[Phase 2: Templates](phase_2_template_migration.md)** | ✅ Detailed refactoring for all page templates, preserving custom JS logic. |
| **[Phase 3: HTMX](phase_3_htmx_cleanup.md)** | ⏳ Strategy for replacing JS polling with HTMX (deferred — requires backend changes). |

---

## 2. Current Stack & Architecture

| Layer | Current | Target |
|---|---|---|
| **Server** | Flask + Jinja2 | *No change* |
| **CSS Framework** | ~~Bootstrap 4.0.0~~ → Halfmoon v2.0.2 | ✅ Done |
| **JS Framework** | ~~jQuery + BS4 JS~~ → Bootstrap 5.3 bundle | ✅ Done (jQuery retained for DataTables) |
| **Icons** | ~~Bootstrap Icons 1.10.5~~ → Phosphor Icons | ✅ Done |
| **Interactivity** | Custom `fetch`/polling JS | HTMX (Phase 3 — deferred) |

---

## 3. Interaction Patterns (Crucial for Migration)

The application uses a mix of interaction patterns that must be preserved:

- **Full Page Reloads**: Login, Profile Updates, Dataset Selection (via redirect).
- **Dynamic Updates**: Progress bars (polled), Log streaming (SSE), Preprocessing wizard steps.
- **Legacy Plugins**: DataTables relies on jQuery.

**Rule**: Do not convert full page reloads to Single Page App (SPA) behavior in Phase 1 or 2.

---

## 4. Verification Checklist

Refer to individual phase guides for specific verification steps.
- [x] Phase 1 Complete — `base.html` + `app.css` Deep Cipher foundation
- [x] Phase 2 Complete — All 9 templates migrated (BS4→BS5 attributes, classes, JS APIs)
- [x] Phase 2.5 Complete — Dark mode polish, hero redesign, pill recoloring, sign-up fix
- [ ] Phase 3 — HTMX cleanup (deferred, requires backend `/progress` endpoint changes)
