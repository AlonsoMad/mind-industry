# Frontend Modernization Guide — MIND Application

> **Audience**: Coding agents and developers tasked with refactoring the MIND frontend.
> **Scope**: Upgrade from Bootstrap 4 to Halfmoon v2 (Bootstrap 5-compatible), add HTMX, modernize typography and icons, clean up legacy JS.

---

## 1. Current Stack & Architecture

| Layer | Current | Target |
|---|---|---|
| **Server** | Flask + Jinja2 | *No change* |
| **CSS Framework** | Bootstrap 4.0.0 (CDN) | Halfmoon v2.0.2 (Bootstrap 5 superset) |
| **JS Framework** | jQuery 3.7.1 + Bootstrap 4 JS + Popper 1.x | Bootstrap 5.3 JS bundle (includes Popper 2) |
| **Data Tables** | DataTables 1.13.6 + `dataTables.bootstrap4` | DataTables + `dataTables.bootstrap5` |
| **Icons** | Bootstrap Icons 1.10.5 | Phosphor Icons |
| **Typography** | Browser defaults | Inter (Google Fonts) |
| **Interactivity** | Custom `fetch`/polling JS | HTMX (progressive enhancement) |
| **Visualization** | D3.js v7 | *No change* |

### Template Inventory

| Template | Lines | Complexity | Key BS4 Patterns |
|---|---|---|---|
| `base.html` | 291 | High | Navbar, CDN links, progress bar polling, alerts |
| `home.html` | 237 | Medium | Hero section, custom pill navigation |
| `detection.html` | 2497 | Very High | Modals, D3 viz, DataTables, SSE, XLSX export |
| `preprocessing.html` | 1279 | High | Multi-step wizard modals, slide navigation |
| `detection_results.html` | 83 | Low | Accordions, links |
| `datasets.html` | 87 | Low | Cards, accordions, tables |
| `profile.html` | 222 | Medium | Forms, drag-and-drop file upload |
| `login.html` | 19 | Low | Login form |
| `sign_up.html` | 27 | Low | Sign-up form |
| `about_us.html` | 31 | Low | Static content |
| `topic_view.html` | 1 | None | Empty file |

---

## 2. CDN & Asset Configuration

### Remove (from `base.html`)

```html
<!-- REMOVE these -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" />
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap4.min.css" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">

<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
```

### Add

```html
<!-- Halfmoon v2 (Bootstrap 5 superset) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/halfmoon@2.0.2/css/halfmoon.min.css" />

<!-- Bootstrap 5.3 JS bundle (includes Popper 2) -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

<!-- DataTables Bootstrap 5 styling -->
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" />

<!-- Google Fonts — Inter -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<!-- Phosphor Icons -->
<script src="https://unpkg.com/@phosphor-icons/web"></script>

<!-- HTMX -->
<script src="https://unpkg.com/htmx.org@1.9.12"></script>
```

### Keep (retain jQuery for DataTables)

```html
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
```

> [!IMPORTANT]
> jQuery must stay until DataTables jQuery dependency is removed. Target removal in a later phase.

### Root CSS Variables (add to `base.html` `<style>` or a new `static/css/app.css`)

```css
:root {
  --font-family-base: 'Inter', system-ui, -apple-system, sans-serif;
}
body {
  font-family: var(--font-family-base);
}
```

---

## 3. Bootstrap 4 → 5 Class Migration Reference

### 3.1 Spacing Utilities

| BS4 | BS5 | Notes |
|---|---|---|
| `ml-*` | `ms-*` | margin-start (LTR = left) |
| `mr-*` | `me-*` | margin-end (LTR = right) |
| `pl-*` | `ps-*` | padding-start |
| `pr-*` | `pe-*` | padding-end |
| `no-gutters` | `g-0` | Grid gutters |

### 3.2 Data Attributes

| BS4 | BS5 |
|---|---|
| `data-toggle` | `data-bs-toggle` |
| `data-target` | `data-bs-target` |
| `data-dismiss` | `data-bs-dismiss` |
| `data-parent` | `data-bs-parent` |
| `data-ride` | `data-bs-ride` |
| `data-slide` | `data-bs-slide` |
| `data-slide-to` | `data-bs-slide-to` |
| `data-placement` | `data-bs-placement` |

### 3.3 Components & Utilities

| BS4 | BS5 | Notes |
|---|---|---|
| `form-group` | *removed* | Use `mb-3` for spacing |
| `form-row` | `row` | |
| `form-inline` | *removed* | Use flex utilities |
| `custom-control` | `form-check` | |
| `custom-control-input` | `form-check-input` | |
| `custom-control-label` | `form-check-label` | |
| `custom-switch` | `form-switch` | Add to parent `form-check` |
| `sr-only` | `visually-hidden` | |
| `close` (button class) | `btn-close` | BS5 close button has no inner `<span>×</span>` — it uses a CSS background |
| `thead-light` | *removed* | Use `table-light` on `<thead>` |
| `btn-block` | `d-block w-100` | |
| `badge-*` (colors) | `bg-*` + text utility | e.g. `bg-primary text-white` |
| `badge-pill` | `rounded-pill` | |
| `text-left` | `text-start` | |
| `text-right` | `text-end` | |
| `border-left` | `border-start` | |
| `border-right` | `border-end` | |
| `font-weight-*` | `fw-*` | |
| `font-style-*` | `fst-*` | |
| `rounded-lg` | `rounded-3` | |
| `rounded-sm` | `rounded-1` | |
| `jumbotron` | *removed* | Recreate with `p-5 bg-body-tertiary rounded-3` |
| `media` | *removed* | Use `d-flex` layout |
| `input-group-append/prepend` | *removed* | Children are direct in `.input-group` |

### 3.4 Close Button Pattern

```html
<!-- BS4 -->
<button type="button" class="close" data-dismiss="modal" aria-label="Close">
  <span aria-hidden="true">&times;</span>
</button>

<!-- BS5 -->
<button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
```

### 3.5 Navbar Changes

- Active class goes on the `<a>` link, not parent `<li>`
- `data-toggle="collapse"` → `data-bs-toggle="collapse"`
- `data-target="#navbarNav"` → `data-bs-target="#navbarNav"`

---

## 4. File-by-File Refactoring Tasks

### 4.1 `base.html` (Priority: 🔴 Critical — do this FIRST)

This is the foundation. All other templates inherit from it.

**CDN swap** (see §2 above)

**Navbar refactoring:**
```diff
- <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav">
+ <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">

- <ul class="navbar-nav ml-auto">
+ <ul class="navbar-nav ms-auto">
```

**Close button in alert dismissal:**
```diff
- <button type="button" class="close" data-dismiss="alert" aria-label="Close">
-   <span aria-hidden="true">&times;</span>
- </button>
+ <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
```

**Progress bar polling:** Keep the existing JS for now. Mark for HTMX replacement in Phase 3.

**Icon migration:** Replace `bi bi-*` classes with Phosphor equivalents:
```diff
- <i class="bi bi-house-door"></i>
+ <i class="ph ph-house"></i>

- <i class="bi bi-gear"></i>
+ <i class="ph ph-gear"></i>

- <i class="bi bi-box-arrow-right"></i>
+ <i class="ph ph-sign-out"></i>
```

---

### 4.2 `home.html` (Priority: 🟡 Medium)

**Jumbotron removal:**
The hero section uses the `jumbotron` class, which is removed in BS5.
```diff
- <div class="jumbotron">
+ <div class="p-5 mb-4 bg-body-tertiary rounded-3">
```

**Custom pills navigation:** The home page has custom-styled pills with inline CSS. These should be preserved but adapted:
- Replace any `ml-*`/`mr-*` with `ms-*`/`me-*`
- The custom `.pill` CSS can remain as-is

---

### 4.3 `detection.html` (Priority: 🔴 Critical — highest complexity)

This is the largest template (2497 lines) with multiple UI patterns.

**Modals (5 instances):** All modals need attribute updates.
```diff
<!-- Config Pipeline Modal -->
- data-toggle="modal" data-target="#configPipelineModalLabel"
+ data-bs-toggle="modal" data-bs-target="#configPipelineModalLabel"

<!-- All close buttons -->
- <button type="button" class="close" data-dismiss="modal" aria-label="Close">
-   <span aria-hidden="true">&times;</span>
- </button>
+ <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>

<!-- Footer close buttons -->
- <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
+ <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
```

**Modal locations in file:**
- `#configPipelineModalLabel` — line ~1651
- `#logModal` — line ~1824
- `#docModal` — line ~986
- `#exitModal` — custom (non-Bootstrap), no change needed

**DataTables styling:**
```diff
- <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap4.min.css">
+ <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
```

**Table header:**
```diff
- <thead class="thead-light">
+ <thead class="table-light">
```

**Screen reader text:**
```diff
- <span class="sr-only">Loading...</span>
+ <span class="visually-hidden">Loading...</span>
```

**Form groups:**
```diff
- <div class="form-group mb-3 d-flex align-items-center">
+ <div class="mb-3 d-flex align-items-center">
```

**Dropdown data attributes:**
```diff
- data-toggle="dropdown"
+ data-bs-toggle="dropdown"
```

**Tab navigation:**
```diff
- data-toggle="tab"
+ data-bs-toggle="tab"

- data-toggle="modal" data-target="#logModal"
+ data-bs-toggle="modal" data-bs-target="#logModal"
```

**Spacing classes (scan entire file):**
```diff
- class="mr-2"
+ class="me-2"

- class="ml-2"
+ class="ms-2"
```

**D3.js visualization:** No changes needed — D3 renders to SVG and doesn't depend on Bootstrap classes.

**jQuery Bootstrap API calls (important):**
The file uses jQuery-style Bootstrap API calls that need updating to vanilla BS5 API:
```diff
- $('#logModal').on('shown.bs.modal', function () { ... });
+ const logModal = document.getElementById('logModal');
+ logModal.addEventListener('shown.bs.modal', function () { ... });

- $('#docModal').modal('show');
+ const docModal = new bootstrap.Modal(document.getElementById('docModal'));
+ docModal.show();
```

> [!WARNING]
> The `detection.html` file heavily uses `$(document).ready()` and jQuery event delegation (`$('body').on(...)`). These must be kept functional until jQuery is fully removed. The BS5 JS bundle works independently of jQuery, but DataTables still needs it.

---

### 4.4 `preprocessing.html` (Priority: 🟠 High)

**Step wizard modals:**
```diff
- data-toggle="modal" data-target="#stepModal"
+ data-bs-toggle="modal" data-bs-target="#stepModal"
```

**Close buttons:** Same pattern as detection.html.

**Collapse component (details toggles):**
The file uses jQuery collapse API:
```javascript
$(collapseEl).collapse('show');
$(collapseEl).collapse('hide');
$(collapseEl).hasClass('show');
```
These should be updated to BS5 vanilla API:
```javascript
const bsCollapse = new bootstrap.Collapse(collapseEl, { toggle: false });
bsCollapse.show();
bsCollapse.hide();
collapseEl.classList.contains('show');
```

**Modal reset event:**
```diff
- $('#stepModal').on('hidden.bs.modal', function () { ... });
+ document.getElementById('stepModal').addEventListener('hidden.bs.modal', function () { ... });
```

**Details toggle data attributes:**
```diff
- <div class="details-toggle" data-target="#detailsSlide1">
```
This uses a custom `data-target` (not Bootstrap's). **No namespace change needed** for custom attributes.

---

### 4.5 `detection_results.html` (Priority: 🟢 Low)

- Replace any `data-toggle` with `data-bs-toggle`
- Update accordion pattern if Bootstrap accordion is used

---

### 4.6 `datasets.html` (Priority: 🟢 Low)

- Replace `thead-light` with `table-light` if tables are present
- Update accordion `data-toggle` attributes

---

### 4.7 `profile.html` (Priority: 🟡 Medium)

**Form elements:**
```diff
- <div class="form-group">
+ <div class="mb-3">

- <label for="...">
+ <label for="..." class="form-label">
```

**Custom file input (drag-and-drop):** Review the drag-and-drop area — it uses custom JS and should work with BS5. Verify styling after CDN swap.

---

### 4.8 `login.html` & `sign_up.html` (Priority: 🟢 Low)

```diff
- <div class="form-group">
+ <div class="mb-3">

- <label>
+ <label class="form-label">
```

---

### 4.9 `about_us.html` (Priority: 🟢 Low)

Minimal changes. Just ensure link styles work with Halfmoon.

---

## 5. Halfmoon v2 Dark Mode Integration

Halfmoon v2 provides built-in dark mode via a `data-bs-theme` attribute on `<html>` or `<body>`.

### Enable dark mode toggle (add to `base.html` navbar):

```html
<button id="theme-toggle" class="btn btn-sm btn-outline-secondary" onclick="toggleTheme()">
  <i class="ph ph-moon"></i>
</button>

<script>
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-bs-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-bs-theme', next);
  localStorage.setItem('theme', next);
}

// Apply saved theme on load
(function() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-bs-theme', saved);
})();
</script>
```

### CSS Variable Overrides for "Sci-fi" Look

Halfmoon uses CSS custom properties. Override in `static/css/app.css`:

```css
[data-bs-theme="dark"] {
  --bs-body-bg: #0f1117;
  --bs-body-color: #e0e6ed;
  --bs-primary: #6366f1;          /* Indigo accent */
  --bs-primary-rgb: 99, 102, 241;
  --bs-card-bg: #1a1d28;
  --bs-border-color: #2d3148;
}

[data-bs-theme="light"] {
  --bs-body-bg: #f8f9fb;
  --bs-body-color: #1f2937;
  --bs-primary: #4f46e5;
  --bs-primary-rgb: 79, 70, 229;
}
```

---

## 6. HTMX Progressive Enhancement (Phase 3)

### Candidate patterns for HTMX replacement:

| Current Pattern | File | HTMX Replacement |
|---|---|---|
| Progress bar polling (`setInterval` + `fetch`) | `base.html` | `hx-get="/progress" hx-trigger="every 2s" hx-swap="innerHTML"` |
| Pipeline status polling | `detection.html` | `hx-get="/pipeline_status" hx-trigger="every 2s" hx-swap="outerHTML"` |
| Dataset selection POST → redirect | `detection.html` | `hx-post="/detection_topickeys" hx-target="#main-content"` |
| Log streaming (SSE) | `detection.html` | `hx-ext="sse" sse-connect="/stream_detection" sse-swap="message"` |

### Example: Progress Bar with HTMX

```html
<!-- Replace manual polling with HTMX -->
<div id="progress-container"
     hx-get="/progress"
     hx-trigger="every 2s"
     hx-swap="innerHTML">
  <!-- Server returns updated progress bar HTML -->
</div>
```

> [!NOTE]
> HTMX adoption is optional in Phase 1 & 2. Prioritize CDN swap and class migration first.

---

## 7. Phased Execution Plan

### Phase 1 — Foundation (est. 2-4 hours)

- [ ] Swap CDN links in `base.html` (§2)
- [ ] Add Inter font and Phosphor Icons
- [ ] Add HTMX script tag
- [ ] Create `static/css/app.css` with CSS variables (§5)
- [ ] Update navbar classes and data attributes
- [ ] Update alert dismiss buttons
- [ ] Add dark mode toggle
- [ ] **Test**: Verify all pages load without console errors

### Phase 2 — Template Migration (est. 4-8 hours)

- [ ] `detection.html` — modals, dropdowns, tables, forms (§4.3)
- [ ] `preprocessing.html` — wizard modals, collapse API (§4.4)
- [ ] `home.html` — jumbotron, pills (§4.2)
- [ ] `profile.html` — forms (§4.7)
- [ ] `login.html` / `sign_up.html` — forms (§4.8)
- [ ] `datasets.html` / `detection_results.html` — minor updates (§4.5, §4.6)
- [ ] `about_us.html` — verify styling (§4.9)
- [ ] Global find-replace: `ml-` → `ms-`, `mr-` → `me-`
- [ ] Global find-replace: `data-toggle` → `data-bs-toggle`, `data-target` → `data-bs-target`, `data-dismiss` → `data-bs-dismiss`
- [ ] **Test**: Every modal, dropdown, accordion, form submission, and table

### Phase 3 — Polish & HTMX (est. 2-4 hours)

- [ ] Replace progress bar polling with HTMX
- [ ] Replace pipeline status polling with HTMX
- [ ] Evaluate SSE log streaming with HTMX SSE extension
- [ ] Remove unused custom CSS that Halfmoon handles
- [ ] Fine-tune dark mode colors and transitions
- [ ] **Test**: Full end-to-end workflow

---

## 8. Automated Find-Replace Commands

Run these from the `app/frontend/templates/` directory:

```bash
# Spacing utilities
sed -i 's/class="ml-/class="ms-/g; s/ ml-/ ms-/g' *.html
sed -i 's/class="mr-/class="me-/g; s/ mr-/ me-/g' *.html
sed -i 's/class="pl-/class="ps-/g; s/ pl-/ ps-/g' *.html
sed -i 's/class="pr-/class="pe-/g; s/ pr-/ me-/g' *.html

# Data attributes
sed -i 's/data-toggle=/data-bs-toggle=/g' *.html
sed -i 's/data-target=/data-bs-target=/g' *.html
sed -i 's/data-dismiss=/data-bs-dismiss=/g' *.html
sed -i 's/data-parent=/data-bs-parent=/g' *.html

# Removed classes
sed -i 's/thead-light/table-light/g' *.html
sed -i 's/sr-only/visually-hidden/g' *.html
```

> [!CAUTION]
> **Do NOT blindly run sed on `data-target=`** inside `preprocessing.html` — it uses a custom `data-target` attribute for detail toggles (not Bootstrap). Inspect each occurrence manually in that file before replacing. The custom `data-target` on `.details-toggle` elements should remain unchanged.

---

## 9. Verification Checklist

### Visual Checks
- [ ] Navbar renders correctly (light & dark mode)
- [ ] All modals open/close properly
- [ ] All dropdowns function
- [ ] All accordion/collapse elements toggle
- [ ] DataTables render with proper BS5 styling
- [ ] D3.js topic visualizations render in detection page
- [ ] Forms submit correctly (login, signup, profile, preprocessing, detection config)
- [ ] Dark mode toggle persists across page navigation
- [ ] Progress bars animate correctly

### Console Checks
- [ ] No `404` errors for removed CDN assets
- [ ] No Bootstrap JS errors about missing jQuery
- [ ] No DataTables initialization errors
- [ ] No D3.js rendering errors

### Functional Checks
- [ ] Login/Logout flow
- [ ] File upload in profile
- [ ] Preprocessing wizard (all 3 steps + slides)
- [ ] Detection pipeline config → run → results
- [ ] XLSX export/download
- [ ] SSE log streaming

---

## 10. UX/UI Improvement Suggestions

### Quick Wins (implement during Phase 2)

1. **Consistent card styling**: Use Halfmoon's card component with subtle shadows for all content sections
2. **Loading states**: Replace `alert()` calls with toast notifications using BS5 Toast component
3. **Form validation**: Use BS5 native form validation (`was-validated`, `is-invalid` classes)
4. **Responsive tables**: Wrap all tables in `.table-responsive`
5. **Breadcrumbs**: Add breadcrumb navigation for deeper pages (detection → results → topic view)

### Visual Enhancements

1. **Color palette**: Move from Bootstrap's default colors to a curated palette:
   - Primary: `#4f46e5` (Indigo)
   - Success: `#10b981` (Emerald)
   - Warning: `#f59e0b` (Amber)
   - Danger: `#ef4444` (Red)
   - Info: `#06b6d4` (Cyan)

2. **Transitions**: Add `transition: all 0.2s ease` to interactive elements

3. **Typography hierarchy**: Use Inter weight variations:
   - Headings: `font-weight: 600`
   - Body: `font-weight: 400`
   - Captions: `font-weight: 300`

---

## 11. Best Practices for Agents

1. **One template at a time** — Complete and test each template before moving to the next
2. **`base.html` first** — All templates extend it; breaking it breaks everything
3. **Preserve custom JS** — Don't refactor JavaScript logic during CSS migration
4. **Test modals explicitly** — They are the most fragile component during BS4→5 migration
5. **Keep inline styles** — Don't try to extract inline styles to CSS files during this migration
6. **Don't touch D3.js** — The visualization code is independent of Bootstrap
7. **Commit after each template** — Enable easy rollback if something breaks
8. **Preserve `data-no-warning`** — Custom attributes on detection page links are used for exit-modal logic
