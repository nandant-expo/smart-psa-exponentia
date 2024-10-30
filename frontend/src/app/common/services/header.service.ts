import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HeaderService {
  isMobHeaderVisibleSubject = new BehaviorSubject<boolean>(false)

  exportSubject = new BehaviorSubject<boolean>(false)
  previewtSubject = new BehaviorSubject<boolean>(false)
  sidebarSubject = new BehaviorSubject<boolean>(false)

  isExportSubject = new BehaviorSubject<boolean>(false)
  citationSubject = new BehaviorSubject<boolean>(false)

  constructor() { }

  setIsExportSubject = (value: boolean) => {
    this.isExportSubject.next(value)
  }

  setCitationSubject = (value: boolean) => {
    this.citationSubject.next(value)
  }

  setIsExportIconVisible = (value: boolean) => {
    this.exportSubject.next(value)
  }

  setIsPreviewIconVisible = (value: boolean) => {
    this.previewtSubject.next(value)
  }

  setSidebarVisible = (value: boolean) => {
    this.sidebarSubject.next(value)
  }

  setIsMobileHeaderVisibleSubject = (value: boolean) => {
    this.isMobHeaderVisibleSubject.next(value);
  }
}
