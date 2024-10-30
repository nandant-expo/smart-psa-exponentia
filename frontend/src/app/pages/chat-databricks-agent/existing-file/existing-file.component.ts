import { Component, ElementRef, ViewChild } from '@angular/core';
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';
import { ApiService } from '../../../common/services/api.service';

export interface FileData {
  ppt_name: string;
  title: string;
  tags: string;
  created_on: string;
}

@Component({
  selector: 'app-existing-file',
  templateUrl: './existing-file.component.html',
  styleUrls: ['./existing-file.component.scss']
})
export class ExistingFileComponent {
  @ViewChild('dt1', { read: ElementRef, static: false }) dt1!: ElementRef<HTMLElement>;
  loading: boolean = true;
  fileData!: FileData[];
  serachStr:string  = ''
  selectedFile: any = null;
  constructor(
    public activeModal: NgbActiveModal,
    private apiService: ApiService ){}

  ngOnInit(): void {
    this.getAllFiles()
  }

  getAllFiles(){
    this.loading = true;
    this.apiService.getAllDatabricksTables().subscribe((d: any)=>{
      this.fileData = d.data;
      this.loading = false;
    },(err)=>{
      this.loading = false;
    })
  }  

  getDateFormated(utcdate: any){
    return this.apiService.getDateFormated(utcdate)
  }

  searchItem(e: any){
    this.serachStr = e.target.value;
    this.selectedFile = null;
  }

  selectFile(file: any){
    this.selectedFile = file;
  }

  closeModal(reason: any){
    this.activeModal.dismiss(reason)
  }
}
