import { Inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';
import { Router, NavigationEnd } from '@angular/router';
import { ApiService } from './api.service';
import { BehaviorSubject } from 'rxjs';
import { MsalService, MsalBroadcastService, MSAL_GUARD_CONFIG, MsalGuardConfiguration } from '@azure/msal-angular';
import { AuthenticationResult, InteractionRequiredAuthError, InteractionStatus, InteractionType, PopupRequest, RedirectRequest } from '@azure/msal-browser';


@Injectable({
  providedIn: 'root'
})
export class AuthService {
  currentRoutepath: any;
  exceptionRoutes = ['/chat','/content/new','/content/edit'];
  verifiedUserData: BehaviorSubject<any> = new BehaviorSubject<any>({});
  constructor(private http: HttpClient,private router: Router, 
    private apiService: ApiService,
    @Inject(MSAL_GUARD_CONFIG) private msalGuardConfig: MsalGuardConfiguration,
  private authService: MsalService) {
    this.router.events.subscribe((event) => {
      if (event instanceof NavigationEnd) {
        this.currentRoutepath = event.url;
      }
    }
    );
  }
  
  userLogin(formData: any){
    return this.http.post(environment.apiendpoint + 'login', formData, { responseType: 'json' });
  }

  validateUser(formData: any){
    return this.http.post(environment.apiendpoint + 'validate-user', formData, { responseType: 'json' });
  }

  verifyEmail(formData: any){
    return this.http.post(environment.apiendpoint + 'verify-email', formData, { responseType: 'json' });
  }

  uniqueWorkspace(workspace_name: any){
    return this.http.get(environment.apiendpoint + 'verify-workspace?workspace_name=' + workspace_name, { responseType: 'json' });
  }

  workspaceRegistration(formData: any){
    return this.http.post(environment.apiendpoint + 'signup', formData, { responseType: 'json' });
  }

  setauthParams(d: any,flag: any){
    localStorage.setItem('role',d.role);
    localStorage.setItem('displayname',d.displayname);
    localStorage.setItem('username',d.username);
    localStorage.setItem('access_token',d.access_token);
    localStorage.setItem('refresh_token',d.refresh_token);
    if(flag == 0){
      this.syncSPUserDoc()
    }else{
      this.router.navigate(['/']);
    }

  }

  async syncSPUserDoc(){
    const accessToken = await this.getSlientAccessToken()
    this.apiService.syncSPUserDoc({personal_access_token:accessToken}).subscribe((users: any)=> {
      this.router.navigate(['/']);
    },err=>{
      this.router.navigate(['/']);
    })
  }
  
  getRole(){
    return localStorage.getItem('role');
  }

  getUserName(){
    return localStorage.getItem('username');
  }

  getDisplayName(){
    return localStorage.getItem('displayname');
  }
  
  getChatRoutes(){ 
    let counter = 0 
    this.exceptionRoutes.filter((r: any) => {
      if(this.currentRoutepath.includes(r)){
        counter++;
      }
    });
    return counter;
  }
  
  logout(){
    if (this.authService.instance.getActiveAccount() !== null &&             
    this.authService.instance.getAllAccounts().length > 0) {
      localStorage.clear();
      this.authService.logout()
    }else{
      localStorage.clear();
      this.router.navigate(['/auth/login']);
    }
  }

  async getSlientAccessToken(){
    if (this.msalGuardConfig.authRequest) {
    let user = this.authService.instance.getActiveAccount();
    var currentAccount = this.authService.instance.getAccountByUsername(user?.username as any);
    var silentRequest = {
        scopes: ["user.read"],
        account: currentAccount,
        forceRefresh: false
      };

    var request = {
        scopes: ["user.read"],
        loginHint: currentAccount?.username // For v1 endpoints, use upn from idToken claims
    };
    const tokenResponse = await this.authService.instance.acquireTokenSilent(silentRequest as any).catch((error: any) => {
      if (error instanceof InteractionRequiredAuthError) {
          // fallback to interaction when silent call fails
          return this.authService.instance.acquireTokenRedirect(request)
      }else{
        console.log(error)
        return
      }
    });
    if(tokenResponse){
      // console.log(tokenResponse)
      return tokenResponse.accessToken
    }else{
      return null;
    }
    }else{
      return null
    }
  }
}
