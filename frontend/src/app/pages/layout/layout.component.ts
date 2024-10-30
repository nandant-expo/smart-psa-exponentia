import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { MenuItem } from 'primeng/api';
import { AuthService } from '../../common/services/auth.service';
import { OverlayPanel } from 'primeng/overlaypanel';
import { HeaderService } from 'src/app/common/services/header.service';
import { Subscription } from 'rxjs';
import { ApiService } from 'src/app/common/services/api.service';

@Component({
  selector: 'app-layout',
  templateUrl: './layout.component.html',
  styleUrls: ['./layout.component.scss']
})
export class LayoutComponent implements OnInit, OnDestroy {
  showNavMenu = 'none';
  currentYear = new Date().getFullYear();
  baritems!: MenuItem[];
  userName: any;
  userRole: any;
  isExportIconVisible = false;
  isPreviewIconVisible = false;
  isMobHeaderVisibleSubscription: Subscription = new Subscription();
  exportSubscription: Subscription = new Subscription();
  headerSubscription: Subscription = new Subscription();
  previewSubscription: Subscription = new Subscription();
  @ViewChild('op', { read: ElementRef }) op!: ElementRef;

  showDiv: boolean = true;
  
  constructor (
    private router: Router,
    private authService: AuthService,
    private headerService: HeaderService,
    private apiService: ApiService,
  ) {}
  
  ngOnInit(){
    const role = this.apiService.getRole();    
    this.userRole = localStorage.getItem('role')
    this.userName = localStorage.getItem('displayname')

    if (role === 'cxo') {
      this.baritems = [
        {
          label: 'SDA',
          command: () => { this.router.navigate(['/structure-data-analysis']); }
        },
        {
          label: 'Dashboard',
          command: () => { this.router.navigate(['/dashboard']); }
        },
        {
          label: 'Table',
          command: () => { this.router.navigate(['/table']); }
        },
        {
          label: 'Chat',
          command: () => { this.router.navigate(['/chat']); }
        },
        {
          label: 'Contract Compliance',
          command: () => { this.router.navigate(['/compliance']); }
        }
      ]
    } else if (role === 'category_manager') {
      this.baritems = [
        {
          label: 'SDA',
          command: () => { this.router.navigate(['/structure-data-analysis']); }
        },
        {
          label: 'Dashboard',
          command: () => { this.router.navigate(['/dashboard']); }
        }
      ]
    } else {
      // if role === 'contract_manager'
      this.baritems = [
        {
          label: 'Dashboard',
          command: () => { this.router.navigate(['/dashboard']); }
        },
        {
          label: 'Table',
          command: () => { this.router.navigate(['/table']); }
        },
        {
          label: 'Chat',
          command: () => { this.router.navigate(['/chat']); }
        }
      ]
    }
    
    this.checkForHeader();
    this.checkForPreview();
    this.checkForExport();
  }

  routeToHome(){
    return '/dashboard'
  }
  checkForExport = () => {
    this.exportSubscription = this.headerService.exportSubject.subscribe(params => {
      this.isExportIconVisible = params
    })
  }

  checkForHeader = () => {
    this.isMobHeaderVisibleSubscription =
      this.headerService.isMobHeaderVisibleSubject.subscribe((value) => {
      this.showDiv = value
    })
  }

  checkForPreview = () => {
    this.previewSubscription = this.headerService.previewtSubject.subscribe(params => {
      this.isPreviewIconVisible = params
    })
  }
  
  showModal(){
    if(this.showNavMenu == 'none'){
      this.showNavMenu = 'block';
    }else{
      this.showNavMenu = 'none';
    }
  }

  logout(){
    this.authService.logout();
  }

  navigateDefinedRoute(r: any,op: OverlayPanel){
    op.hide()
    this.router.navigate([r]);
  }

  onSearchMobile = (op: OverlayPanel) => {
    this.headerService.setSidebarVisible(false);
    op.hide()
    this.router.navigate(['/']);
  }

  onNewChatMobile = (op: OverlayPanel) => {
    this.headerService.setSidebarVisible(false);
    op.hide()
    this.router.navigate(['/chat']);
  }

  onSettingsMobile = (op: OverlayPanel) => {
    this.headerService.setSidebarVisible(false);
    op.hide()
    this.router.navigate(['/settings']);
  }
  
  getChatRoutes(){
    return this.authService.getChatRoutes();
  }

  openChatSidebar = () => {
    this.headerService.setSidebarVisible(true)
  }

  onExportMobile = () => {
    this.headerService.setIsExportSubject(true);
  }

  onCitationMobile = () => {
    this.headerService.setCitationSubject(true)
  }

  ngOnDestroy(): void {
    this.isMobHeaderVisibleSubscription.unsubscribe();
    this.exportSubscription.unsubscribe();
    this.headerSubscription.unsubscribe();
    this.previewSubscription.unsubscribe();
  }
}
