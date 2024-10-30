import { AfterViewInit, Component, ElementRef, EventEmitter, OnInit, Output, ViewChild } from '@angular/core';
import { PrimeNGConfig } from 'primeng/api';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit, OnInit {
  constructor(private primengConfig: PrimeNGConfig) {
  }

  ngOnInit() {
      this.primengConfig.ripple = true;
  }

  ngAfterViewInit(): void {
  }
}
