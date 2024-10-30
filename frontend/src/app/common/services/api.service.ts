import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from 'src/environments/environment';
import { BehaviorSubject } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class ApiService {

  category: BehaviorSubject<string> = new BehaviorSubject<string>('');
  breadcumList: BehaviorSubject<any> = new BehaviorSubject<any>(null);
  constructor(private http: HttpClient) { }
  
  getRole(){
    return localStorage.getItem('role');
  }

  getADUsersCount(accessToken: any) {
    const headers = new HttpHeaders({ 'No-Auth': 'True', 'Content-Type':'application/x-www-form-urlencoded','ConsistencyLevel':'eventual', 'Authorization':'Bearer '+accessToken});
    return this.http.get<any>( environment.graphAPIUrl+ 'users/$count',{headers:headers, responseType: 'json'})
  }

  //get maximum 999 Users from Active Directory 
  getADUsers(url: any, accessToken :any) {
    const headers = new HttpHeaders({ 'No-Auth': 'True', 'Content-Type':'application/x-www-form-urlencoded', 'Authorization':'Bearer '+accessToken});
    return this.http.get<any>( url ,{headers:headers, responseType: 'json'})
  }
  
  getPPT(formData: any){
    return this.http.post(environment.apiendpoint + 'ppts',formData , { responseType: 'json' });
  }

  categorizePPT(formData: any){
    return this.http.post(environment.apiendpoint + 'categorize', formData , { responseType: 'json' });
  }
  
  getPPTByCategory(formData: any){
    return this.http.post(environment.apiendpoint + 'category', formData , { responseType: 'json' });
  }

  getPPTBySerachTerm(formData: any){
    return this.http.post(environment.apiendpoint + 'search', formData , { responseType: 'json' });
  }
  
  getPPTBySerachTermAndVintage(formData: any){
    return this.http.post(environment.apiendpoint + 'search_vintage', formData , { responseType: 'json' });
  }

  getSlideBySerachTerm(formData: any){
    return this.http.post(environment.apiendpoint + 'search_slide', formData , { responseType: 'json' });
  }
  
  getSlideBySerachTermAndVintage(formData: any){
    return this.http.post(environment.apiendpoint + 'slide_vintage', formData , { responseType: 'json' });
  }
  
  getHistoryRecommendation(formData: any){
    return this.http.post(environment.apiendpoint + 'search_history', formData , { responseType: 'json' });
  }

  processFile(formData: any){
    return this.http.post(environment.apiendpoint + 'upload', formData, { responseType: 'json' });
  }

  create(formData: any){
    return this.http.post(environment.apiendpoint + 'create', formData, { responseType: 'json' });
  }

  saveSettings(formData: any){
    return this.http.post(environment.apiendpoint + 'update_settings', formData, { responseType: 'json' });
  }

  getSettings(){
    return this.http.get(environment.apiendpoint + 'get_settings', { responseType: 'json' });
  }

  processfirst(formData: any){
    return this.http.post(environment.apiendpoint + 'process1', formData, { responseType: 'json' });
  }
  
  processsecond(formData: any){
    return this.http.post(environment.apiendpoint + 'process2', formData, { responseType: 'json' });
  }

  preview(formData: any){
    return this.http.post(environment.apiendpoint + 'preview', formData, { responseType: 'json' });
  }
  
  getAllFiles(){
    return this.http.get(environment.apiendpoint + 'fetch_all', { responseType: 'json' });
  }

  getMetaData(formData: any){
    return this.http.post(environment.apiendpoint + 'fetch_one', formData, { responseType: 'json' });
  }

  saveMetaData(formData: any){
    return this.http.post(environment.apiendpoint + 'edit', formData, { responseType: 'json' });    
  }

  deleteMetaData(formData: any){
    return this.http.post(environment.apiendpoint + 'delete', formData, { responseType: 'json' });
  }

  getListOfRelevancePPT(formData: any){
    return this.http.post(environment.apiendpoint + 'relevant_ppts', formData , { responseType: 'json' });
  }
  
  getListOfRelevanceSlides(formData: any){
    return this.http.post(environment.apiendpoint + 'relevant_slides', formData , { responseType: 'json' });
  }
  
  getPPTById(formData: any){
    return this.http.post(environment.apiendpoint + 'full_presentation', formData , { responseType: 'json' });
  }

  downloadSlides(formData: any){
    return this.http.post(environment.apiendpoint + 'download', formData , { responseType: 'json' });
  }

  getAllUsers(){
    return this.http.get(environment.apiendpoint + 'users', { responseType: 'json' });
  }

  saveUsers(formData: any){
    return this.http.post(environment.apiendpoint + 'users', formData, { responseType: 'json' });
  }

  updateUsers(formData: any,id: any){
    return this.http.put(environment.apiendpoint + 'users?email='+ id, formData, { responseType: 'json' });
  }

  deleteUser(id: any) {
    return this.http.delete(environment.apiendpoint + 'users?email='+ id, { responseType: 'json' });
  }

  getConnections(){
    return this.http.get(environment.apiendpoint + 'connectorSettings', { responseType: 'json' });
  }

  saveConnections(formData: any){
    return this.http.post(environment.apiendpoint + 'connectorSettings', formData, { responseType: 'json' });
  }

  savePipelines(formData: any){
    return this.http.post(environment.apiendpoint + 'pipelineSettings', formData, { responseType: 'json' });
  }

  deletePipelines(formData: any) {
    return this.http.post(environment.apiendpoint + 'pipelineDelete', formData, { responseType: 'json' });
  }

  syncSPUserDoc(formData: any) {
    return this.http.post(environment.apiendpoint + 'access_sync', formData, { responseType: 'json' });
  }

  testDatabricksConnection(formData: any){
    return this.http.post(environment.apiendpoint + 'databricksConnection', formData, { responseType: 'json' });
  }

  getDatabricksConnection(){
    return this.http.get(environment.apiendpoint + 'databricksConnection', { responseType: 'json' });
  }

  saveDatabricksConnection(formData: any){
    return this.http.patch(environment.apiendpoint + 'databricksConnection', formData, { responseType: 'json' });
  }

  getAllDatabricksTables(){
    return this.http.get(environment.apiendpoint + 'fetch_table', { responseType: 'json' });
  }

  categorizeDocumentDesc(formData: any){
    return this.http.post(environment.apiendpoint + 'categorise_file', formData , { responseType: 'json' });
  }

  // common functions

  capitalizeFirstLetter(string: string) {
    return  string
    .toLowerCase()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  }

  downloadURI(uri: any, name: any) {
    const link = document.createElement('a');
    link.setAttribute('style', 'display: none');
    link.setAttribute('target', '_blank');
    link.setAttribute('href', uri);
    link.setAttribute('download', name);
    document.body.appendChild(link);
    link.click();
    link.remove();
  }
  getDateFormated(utcdate: any){
    let date = new Date(utcdate)
    var newDate = new Date(date.getTime() - date.getTimezoneOffset()*60*1000);
    return newDate
  }

  convertUTCDateToLocalDate(utcDateString: any) {
    // Create a Date object from the UTC string
    const utcDate = new Date(utcDateString);
    // get Local Time Zone
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    // Convert UTC date to local date using toLocaleString() with timeZone option
    const options = { timeZone: timeZone }; // Specify your target time zone
    return utcDate.toLocaleString('en-US', options);
  }

  attachTimestampToFileName(fileName: any) {
    const timestamp = new Date().getTime();
    const parts = fileName.split('.');
    const extension = parts.pop();
    const name = parts.join('.')+'_'+timestamp;
    const updatedFileName = `${name}.${extension}`;

    return {filename:updatedFileName,fileNameWithoutExt:name};
  }

}
