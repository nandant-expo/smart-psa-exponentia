import { hideElementOverflow, ZoomDragHTMLElement } from '.';

/**
 * Makes the Element draggable (makes it movable inside its parent element)
 * @param param.element - The element to be draggable
 * @param param.overflowHidden - If true, hides element when it overflows its parent (defaults to `false`)
 * @param param.DRAGGABLE_CSS_CLASS - Custom CSS class name to be used (optional / internally provided)
 * @return element with `.stopDraggableBehavior()` method to help stop behavior
 * @example
 * // make element draggable
 * const draggableElmnt = makeElementDraggable({ elementId: document.getElementById('my-element') });
 * // now it's not draggable anymore
 * draggableElmnt.stopDraggableBehavior();
 */
export const makeElementDraggable = ({
  element,
  isOverflowHiddenDesired = false,
  DRAGGABLE_CSS_CLASS = 'xyzw-draggable-element',
}: {
  element: HTMLElement;
  isOverflowHiddenDesired?: boolean;
  DRAGGABLE_CSS_CLASS?: string;
}): ZoomDragHTMLElement | null => {
  // log error, if the element is not found, and exit execution
  if (!element) {
    console.error(`Error: makeElementDraggable() needs an element to work`);
    return null;
  }

  // sets "overflow: hidden", if desired
  element = isOverflowHiddenDesired
    ? // tslint:disable-next-line:no-non-null-assertion
      hideElementOverflow({ element })!
    : element;

  // check if necessary CSS is already added to body (to avoid polluting html)
  const isTheRequiredCSSAlreadyInPlace = !!Array.from(
    element.ownerDocument.getElementsByTagName('style')
  ).filter((styleEl) =>
    styleEl?.textContent?.includes(`.${DRAGGABLE_CSS_CLASS}`)
  )?.[0];

  // append the necessary CSS, to body, to enable draggability (if not appended already)
  if (!isTheRequiredCSSAlreadyInPlace) {
    element.ownerDocument.body.appendChild(
      Object.assign(element.ownerDocument.createElement('style'), {
        textContent: `
        .${DRAGGABLE_CSS_CLASS} {
          position: absolute !important;
          cursor: grabbing !important;
        }
        `,
      })
    );
  }

  // add the necessary CSS class, to element, so that it becomes draggable (if not added already)
  if (!element.classList.contains(DRAGGABLE_CSS_CLASS)) {
    element.classList.add(DRAGGABLE_CSS_CLASS);
  }

  // make the element follow the mouse, while pressing mouse button (in other words drags the element)
  element.onmousedown = (mouseDownEvent) => {
    mouseDownEvent.preventDefault();

    element.onmousemove = (mouseMoveEvent) => {
      mouseMoveEvent.preventDefault();

      element.style.left = `${element.offsetLeft + mouseMoveEvent.movementX}px`;
      element.style.top = `${element.offsetTop + mouseMoveEvent.movementY}px`;
    };
  };

    // stops following (in other words, stops dragging)
    element.onclick = (mouseClickEvent) => {
      // console.log("in onclick")
      mouseClickEvent.preventDefault();
  
      element.onmousemove = null;
    };

  // stops following (in other words, stops dragging)
  element.ownerDocument.onmouseup = (mouseUpEvent) => {
    mouseUpEvent.preventDefault();

    element.onmousemove = null;
  };

  // returns a method to help remove behavior from element, if needed
  return Object.assign(element, {
    stopDraggableBehavior() {
      element.classList.remove(DRAGGABLE_CSS_CLASS);
      element.onmousedown = null;
      element.onmousemove = null;
      element.onmouseup = null;
      element.style.left = 'unset';
      element.style.top = 'unset';
      (element as ZoomDragHTMLElement).stopHiddenOverflowBehavior?.();
      return element;
    },
  });
};
