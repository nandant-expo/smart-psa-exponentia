import { Component, OnInit } from '@angular/core';
import { HeaderService } from 'src/app/common/services/header.service';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss']
})
export class MainComponent implements OnInit {
  isChatHistoryVisible = false;
  isMobile = window.innerWidth < 768

  constructor(
    private headerService: HeaderService
  ) {}

  ngOnInit(): void {
    this.checkForSidebar()
  }

  checkForSidebar = () => {
    this.headerService.sidebarSubject.subscribe(params => {
      this.isChatHistoryVisible = params;
    })
  }
}
