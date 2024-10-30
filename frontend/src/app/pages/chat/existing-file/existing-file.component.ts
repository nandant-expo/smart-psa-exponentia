import { Component, ViewChild } from '@angular/core';
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';
import { ApiService } from '../../../common/services/api.service';
import { FormControl } from '@angular/forms';
import { Observable, debounceTime, distinctUntilChanged } from 'rxjs';
import { ConfirmationService, MessageService } from 'primeng/api';
import { Table } from 'primeng/table';

export interface FileData {
  _id: string;
  display_name: string;
  file_type: string;
  ppt_name: string;
  title: string;
  tags: string;
  created_on: string;
  uploaded_on: string;
}

@Component({
  selector: 'app-existing-file',
  templateUrl: './existing-file.component.html',
  styleUrls: ['./existing-file.component.scss'],
  providers: [ConfirmationService, MessageService]
})
export class ExistingFileComponent {
  @ViewChild('dt1',{static:true}) dt1!: Table;
  loading: boolean = true;
  fileData: FileData[] = [];
  filteredFileData: FileData[] = [];
  serachStr:string  = ''
  selectedFile: any = null;
  highLightedIndex = -1;
  searchControl = new FormControl();
  filteredList$!: Observable<any[]>;
  
  constructor(
    public activeModal: NgbActiveModal,
    private apiService: ApiService,
    private messageService: MessageService ){}

  ngOnInit(): void {
    this.getAllFiles()

    this.searchControl.valueChanges.pipe(
      debounceTime(300),
      distinctUntilChanged(),
    ).subscribe((term: string) => {
      this.filterList(term)
    });
  }

  getAllFiles(){
    this.loading = true;
    this.apiService.getAllFiles().subscribe((d: any)=>{
      this.fileData = d.data;
      this.filteredList$ = new Observable<any[]>(observer => {
        observer.next(d.data);
        observer.complete();
      });
      this.filteredFileData = [...this.fileData];
      this.loading = false;
    },(err)=>{
      this.loading = false;
    })
  }  

  getDateFormated(utcdate: any){
    if (!utcdate) return;

    return this.apiService.getDateFormated(utcdate)
  }

  searchItem(e: any){
    this.serachStr = e.target.value;
    this.selectedFile = null;
  }
  onFilter(event: any) { 
    this.dt1.totalRecords = event.filteredValue.length; // count of displayed rows 
  }

  filterList(term: string){
    const filteredList: any = this.fileData.filter((item: any) => {
      return (item?.file_name.toLowerCase().includes(term.toLowerCase()) ||
      item?.file_source?.toLowerCase()?.includes(term.toLowerCase()) ||
      item?.status?.toLowerCase()?.includes(term.toLowerCase()))
  });
    console.log(filteredList)
    this.filteredList$ = new Observable<any[]>(observer => {
      observer.next(filteredList);
      observer.complete();
    });
  }

  onMobFileSelect = (index: number, file: any) => {
    if(file?.status == 'Ready'){
      this.highLightedIndex = index;
      this.selectedFile = file?._id;
    }else{
      this.selectedFile = null;
      this.highLightedIndex = -1;
      this.messageService.add({ severity: 'info', summary: 'File not ready !', detail: 'Please try after the file status has changed to "Ready".' });
    }
    
  }

  selectFile(file: any){
    if(file?.status == 'Ready'){
      this.selectedFile = file._id
    }else{
      this.selectedFile = null;
      this.messageService.add({ severity: 'info', summary: 'File not ready !', detail: 'Please try after the file status has changed to "Ready".' });
    }
  }

  closeModal(reason: any){
    this.activeModal.dismiss(reason)
  }
}
