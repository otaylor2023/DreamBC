function pauseMediaIn(container) {
  container.querySelectorAll("video").forEach((v) => v.pause());
}

function playActiveMediaIn(container) {
  container.querySelectorAll(".example-cell").forEach((cell) => {
    const active = cell.querySelector(".media-stack > [data-ex]:not([hidden])");
    if (active && active.tagName === "VIDEO") {
      active.currentTime = 0;
      active.play().catch(() => {});
    }
  });
  container.querySelectorAll("video").forEach((v) => {
    if (!v.closest(".example-cell") && !v.hasAttribute("hidden")) {
      v.currentTime = 0;
      v.play().catch(() => {});
    }
  });
}

function setExampleInCell(cell, ex) {
  const exKey = String(ex);
  cell.querySelectorAll(".media-stack > [data-ex]").forEach((el) => {
    const isActive = el.dataset.ex === exKey;
    el.toggleAttribute("hidden", !isActive);
    if (el.tagName === "VIDEO") {
      if (isActive) {
        el.currentTime = 0;
        el.play().catch(() => {});
      } else {
        el.pause();
      }
    }
  });
  cell.querySelectorAll(".example-selector button").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.ex === exKey);
  });
}

function initExampleCells() {
  document.querySelectorAll(".example-cell").forEach((cell) => {
    const activeBtn = cell.querySelector(".example-selector button.is-active");
    const initialEx = activeBtn ? activeBtn.dataset.ex : "1";
    setExampleInCell(cell, initialEx);

    cell.querySelectorAll(".example-selector button").forEach((btn) => {
      btn.addEventListener("click", () => {
        setExampleInCell(cell, btn.dataset.ex);
      });
    });
  });
}

function setExampleInGroup(group, ex) {
  const exKey = String(ex);
  group.dataset.activeEx = exKey;
  group.querySelectorAll(".media-stack > [data-ex]").forEach((el) => {
    const isActive = el.dataset.ex === exKey;
    el.toggleAttribute("hidden", !isActive);
    if (el.tagName === "VIDEO") {
      if (isActive) {
        el.currentTime = 0;
        el.play().catch(() => {});
      } else {
        el.pause();
      }
    }
  });
  group.querySelectorAll(":scope > .example-selector button").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.ex === exKey);
  });
}

function initExampleGroups() {
  document.querySelectorAll(".example-group").forEach((group) => {
    const activeBtn = group.querySelector(":scope > .example-selector button.is-active");
    const initialEx = activeBtn ? activeBtn.dataset.ex : (group.dataset.activeEx || "1");
    setExampleInGroup(group, initialEx);

    group.querySelectorAll(":scope > .example-selector button").forEach((btn) => {
      btn.addEventListener("click", () => {
        setExampleInGroup(group, btn.dataset.ex);
      });
    });
  });
}

function setAttrSwitchGroup(group, attr, value) {
  group.querySelectorAll(`.media-stack > [data-${attr}]`).forEach((el) => {
    const isActive = el.dataset[attr] === value;
    el.toggleAttribute("hidden", !isActive);
    if (el.tagName === "VIDEO") {
      if (isActive) {
        el.currentTime = 0;
        el.play().catch(() => {});
      } else {
        el.pause();
      }
    }
  });

  group.querySelectorAll(`[data-switch="${attr}"] button`).forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.value === value);
  });
}

function initAttrSwitchGroups(groupSelector, attr, datasetKey, fallback) {
  document.querySelectorAll(groupSelector).forEach((group) => {
    const initial = group.dataset[datasetKey] || fallback;
    group.dataset[datasetKey] = initial;
    setAttrSwitchGroup(group, attr, initial);

    group.querySelectorAll(`[data-switch="${attr}"] button`).forEach((btn) => {
      btn.addEventListener("click", () => {
        group.dataset[datasetKey] = btn.dataset.value;
        setAttrSwitchGroup(group, attr, btn.dataset.value);
      });
    });
  });
}

function initTaskOnlyGroups() {
  initAttrSwitchGroups(".task-only-group", "task", "activeTask", "cube");
}

function initOutcomeOnlyGroups() {
  initAttrSwitchGroups(".outcome-only-group", "outcome", "activeOutcome", "success");
}

function setSampleGallery(group) {
  const task = group.dataset.activeTask || "cube";
  const outcome = group.dataset.activeOutcome || "success";

  group.querySelectorAll(".media-stack > video").forEach((video) => {
    const isActive = video.dataset.task === task && video.dataset.outcome === outcome;
    video.toggleAttribute("hidden", !isActive);
    if (isActive) {
      video.currentTime = 0;
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  });

  group.querySelectorAll("[data-outcome-label]").forEach((label) => {
    label.toggleAttribute("hidden", label.dataset.outcomeLabel !== outcome);
  });

  group.querySelectorAll('[data-switch="task"] button').forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.value === task);
  });
  group.querySelectorAll('[data-switch="outcome"] button').forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.value === outcome);
  });
}

function initSampleGalleries() {
  document.querySelectorAll(".sample-gallery").forEach((group) => {
    group.dataset.activeTask = group.dataset.activeTask || "cube";
    group.dataset.activeOutcome = group.dataset.activeOutcome || "success";
    setSampleGallery(group);

    group.querySelectorAll('[data-switch="task"] button').forEach((btn) => {
      btn.addEventListener("click", () => {
        group.dataset.activeTask = btn.dataset.value;
        setSampleGallery(group);
      });
    });
    group.querySelectorAll('[data-switch="outcome"] button').forEach((btn) => {
      btn.addEventListener("click", () => {
        group.dataset.activeOutcome = btn.dataset.value;
        setSampleGallery(group);
      });
    });
  });
}

function setActiveTaskInGroup(group, task) {
  const t = task === "tomato" ? "tomato" : "cube";
  group.dataset.activeTask = t;
  const switcher = group.querySelector(".task-switcher-local");
  if (switcher) {
    switcher.querySelectorAll("li").forEach((li) => {
      li.classList.toggle("is-active", li.dataset.task === t);
    });
  }
  group.querySelectorAll(".task-pane").forEach((pane) => {
    if (pane.dataset.task === t) {
      playActiveMediaIn(pane);
    } else {
      pauseMediaIn(pane);
    }
  });
}

function initTaskGroups() {
  document.querySelectorAll(".task-group").forEach((group) => {
    const switcher = group.querySelector(".task-switcher-local");
    if (!switcher) return;

    switcher.querySelectorAll("li").forEach((li) => {
      li.addEventListener("click", (e) => {
        e.preventDefault();
        setActiveTaskInGroup(group, li.dataset.task);
      });
    });

    const initial = group.dataset.activeTask === "tomato" ? "tomato" : "cube";
    setActiveTaskInGroup(group, initial);
  });
}

/** Native controls hidden until the user clicks the video (fixes Safari showing
 *  control chrome on every autoplay clip at load). */
function initClickToShowControls() {
  document.querySelectorAll("video[controls]").forEach((video) => {
    video.removeAttribute("controls");
    video.classList.add("video-click-controls");

    video.addEventListener(
      "click",
      () => {
        if (!video.hasAttribute("controls")) {
          video.setAttribute("controls", "");
        }
      },
      { passive: true }
    );
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initClickToShowControls();
  initExampleCells();
  initExampleGroups();
  initTaskOnlyGroups();
  initOutcomeOnlyGroups();
  initSampleGalleries();
  initTaskGroups();

  if (window.bulmaCarousel) {
    document.querySelectorAll(".carousel").forEach((el) => {
      bulmaCarousel.attach(el, {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        autoplay: false,
      });
    });
  }
});
