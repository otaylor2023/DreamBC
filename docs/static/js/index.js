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

document.addEventListener("DOMContentLoaded", () => {
  initExampleCells();
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
