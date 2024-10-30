import { hideElementOverflow, ZoomDragHTMLElement } from '.';

/**
 * Makes the Element zoomable
 * @param param.element - the element to be zoomable
 * @param param.zoomInBtnId - id of the zoom-in button (defaults to `xyzw-zoom-in-button`)
 * @param param.zoomOutBtnId - id of the zoom-out button (defaults to `xyzw-zoom-out-button`)
 * @param param.steps - steps to zoom-in/zoom-out (defaults to `30`)
 * @param param.CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS - Custom CSS class name to be used (optional / internally provided)
 * @return element with `.stopZoomableBehavior()` method to help stop behavior
 * @example
 * // make element zoomable
 * const zoomableElmnt = makeElementZoomable({ element: document.getElementById('my-element') });
 * // now it's not zoomable anymore
 * zoomableElmnt.stopZoomableBehavior();
 */
export const makeElementZoomable = ({
  element,
  zoomInBtnId = 'xyzw-zoom-in-button',
  zoomOutBtnId = 'xyzw-zoom-out-button',
  // tslint:disable-next-line: no-magic-numbers
  steps = 30,
  CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS = 'xyzw-container',
}: {
  element: HTMLElement;
  zoomInBtnId?: string;
  zoomOutBtnId?: string;
  steps?: number;
  CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS?: string;
}): ZoomDragHTMLElement | null => {
  // log error, if the element is not found, and exit execution
  if (!element) {
    console.error(`Error: makeElementZoomable() needs an element to work`);
    return null;
  }

  // gets the element and, at the same time, add the behavior: hide when element
  // overflows parent, while zooming
  // tslint:disable-next-line:no-non-null-assertion
  element = hideElementOverflow({
    element,
    CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS,
  })!;

  // gets the html elements necessary for the functionality
  const zoomInBtn = element.ownerDocument.getElementById(zoomInBtnId);
  const zoomOutBtn = element.ownerDocument.getElementById(zoomOutBtnId);

  // log error, if the element is not found, and exit execution
  if (!zoomInBtn || !zoomOutBtn) {
    console.error(
      `Error: makeElementZoomable function can't find the elements with the following ids:
      ${!zoomInBtn ? `zoomInBtnId: ${zoomInBtnId}` : ''}
      ${!zoomOutBtn ? `zoomOutBtnId: ${zoomOutBtnId}` : ''}`
    );
    return null;
  }

  // value of the css scale ( 1 represents 100% )
  let scaleValue = 1;

  // convert steps to decimal, to allow percentage calculation
  // tslint:disable-next-line: no-magic-numbers
  steps /= 100;

  // zoom in
  zoomInBtn.onclick = () => {
    scaleValue *= 1 + steps;
    element.style.transform = `scale(${scaleValue})`;
  };

  // zoom out
  zoomOutBtn.onclick = () => {
    scaleValue /= 1 + steps;
    element.style.transform = `scale(${scaleValue})`;
  };

  // zoom in/out with mouse wheel
  element.onwheel = (event) => {
    event.preventDefault();

    if (event.deltaY < 0) {
      scaleValue *= 1 + steps;
    } else {
      scaleValue /= 1 + steps;
    }

    element.style.transform = `scale(${scaleValue})`;
  };

  // returns an element with .stopZoomableBehavior() method to
  // help remove zoomable behavior from element, if needed
  return Object.assign(element, {
    stopZoomableBehavior() {
        scaleValue = 1;
      // zoomInBtn.onclick = null;
      // zoomOutBtn.onclick = null;
      // element.onwheel = null;
      (element as ZoomDragHTMLElement).stopHiddenOverflowBehavior?.();
      element.style.transform = 'scale(1)';
      return element;
    },
  });
};
