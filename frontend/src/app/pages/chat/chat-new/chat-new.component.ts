import { AfterViewInit, Component, ElementRef, OnDestroy, OnInit, Renderer2, ViewChild } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { MessageService } from 'primeng/api';
import { environment } from 'src/environments/environment';
import { forkJoin, interval, Subscription } from 'rxjs';
import { Router } from '@angular/router';
import { ApiService } from 'src/app/common/services/api.service';
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { ExistingFileComponent } from '../existing-file/existing-file.component';
import { BlobStorageService } from 'src/app/common/services/blob-storage.service';
import { HeaderService } from 'src/app/common/services/header.service';

@Component({
  selector: 'app-chat-new',
  templateUrl: './chat-new.component.html',
  styleUrls: ['./chat-new.component.scss'],
  providers: [MessageService]
})
export class ChatNewComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('fileForm') fileForm!: ElementRef;
  userName: any;
  uploadFlag: boolean = false;
  isCitationWindowOpen = false;
  loadingmsg: string = '';
  progress: any = '0%'
  subTimer: any;
  percentCount = 0;
  modalRef: any;
  citationSubjectSubscription = new Subscription();
  unlisten = () => {};

  constructor (
    private router: Router, 
    protected chatService: ChatService,
    private messageService: MessageService,
    protected apiService: ApiService,
    private modalService: NgbModal,
    private blobStorageService: BlobStorageService,
    private elementRef: ElementRef,
    private renderer: Renderer2,
    private headerService: HeaderService,
  ) { }
  ngOnInit(): void {
    this.userName = localStorage.getItem('displayname')

    this.citationSubjectSubscription =
      this.headerService.citationSubject.subscribe(param => {
        this.isCitationWindowOpen = param;
      })

    if(window.innerWidth < 1024) {
      // For MOBILE and above
      this.headerService.setIsPreviewIconVisible(false);
      this.headerService.setIsExportIconVisible(false);
      this.headerService.setIsMobileHeaderVisibleSubject(true)
    } else {
      // For LAPTOP and above
      this.headerService.setIsMobileHeaderVisibleSubject(false)
    }
  }

  ngAfterViewInit(): void {
    this.addClickListener()
  }

  addClickListener = () => {
    const chatPannels = this.elementRef.nativeElement.querySelectorAll('.chat-pannel')
    chatPannels.forEach((cp: any) => {
      this.unlisten = this.renderer.listen(cp, 'click', () => {
        this.handleClick()
      })
    })
  }

  handleClick = () => {
    this.headerService.setSidebarVisible(false)
  }
  
  clickFile(){
    this.fileForm?.nativeElement.click();
  }

  onFileChange(event: any) {
    if (event.target.files) {
      this.loadingmsg = 'Uploading...'
      this.uploadFlag = true;
      let file_name = event.target.files[0].name;
      let uploadFileName: any = this.apiService.attachTimestampToFileName(file_name)
      const blobpath = `${environment.bloburl}/${environment.containerName}/${environment.chatContainerName}/${environment.folderName}/${uploadFileName.filename}`;
      this.progress = '0%';
      this.uploadToBlobStorage(event.target.files[0], uploadFileName.filename).then((d: any) => {
        this.loadingmsg = 'Analyzing...';
        this.progress = '0%';
        this.percentCount = 0;
        this.loaderCounter()
        forkJoin(this.chatService.chatUploadProcess({url:blobpath,display_name: uploadFileName.filename}),this.chatService.chatUploadPreview({url:blobpath,display_name: uploadFileName.fileNameWithoutExt}))
        .subscribe((obj: any)=>{
          const chatDtl: any  = obj[0]
          this.chatService.chatUploadStatus({
            url:blobpath,
            display_name: uploadFileName.fileNameWithoutExt
          }).subscribe((obj: any)=>{
            this.subTimer.unsubscribe()
            this.progress = `100%`;
            this.chatService.newQuery.next({'_id': chatDtl.chat_id, 'title':chatDtl.title, 'updated_on': chatDtl.updated_on,"chat_flow":chatDtl.chat_flow});
            this.redirectPage(chatDtl.chat_id) 
          })
        },(err)=>{
          this.messageService.add({ severity: 'error', summary: 'Error', detail: 'Failed to process file, please try again.' });
          this.uploadFlag = false
        })
      },
        (err) => {
          this.messageService.add({ severity: 'error', summary: 'Error', detail: 'Failed to upload file, please try again.' });
          this.uploadFlag = false
        })
    }
  }



  redirectPage(id: any){
    this.uploadFlag = false;
    this.loadingmsg = '';
    let objId = encodeURI(btoa(JSON.stringify(id)).toString()) 
    let type = encodeURI(btoa(JSON.stringify('local')).toString())
    this.router.navigate(['/chat/c/',objId,type]);
  }

  loaderCounter() {
    this.subTimer = interval(600).subscribe((val) => {
      this.percentCount ++;
      this.progress = this.percentCount + '%';
      if(this.percentCount == 96){
        this.subTimer.unsubscribe()
      }
    })
  }

  uploadToBlobStorage(content: Blob, name: string) {
    return new Promise((resolve) => {
      const blockBlobClient = this.blobStorageService.containerClient().getBlockBlobClient(`${environment.chatContainerName}/${environment.folderName}/${name}`);
      blockBlobClient.uploadData(content, {
        blockSize: 4 * 1024 * 1024, // 4MB block size
        concurrency: 20, // 20 concurrency
        onProgress: (ev) => {
          let percent = Math.round(ev.loadedBytes / content.size * 100)
          this.progress = `${percent}%`;
        }
        , blobHTTPHeaders: { blobContentType: content.type }
      })
        .then((data: any) => {
          resolve({ state: true })
        }).catch((err) => {
          resolve({ state: false })
        })
    })
  }

  chatExistingFileProcess(data: any){
    let formData = {
      file_id: data.id
    }
    this.chatService.chatExistingFileProcess(formData).subscribe((obj: any)=>{
      this.subTimer.unsubscribe()
      this.progress = `100%`;
      this.chatService.newQuery.next({'_id': obj.chat_id, 'title':obj.title, 'updated_on': obj.updated_on,"chat_flow":obj.chat_flow});
      this.redirectPage(obj.chat_id) 
    },err=>{
      this.messageService.add({ severity: 'error', summary: 'Error', detail: 'Failed to upload file, please try again.' });
      this.uploadFlag = false;
    })
  }

  openExistingFileWindow(){
    const isMobile = window.innerWidth < 1024;
    
    this.modalService.open(ExistingFileComponent, {
      backdrop: 'static',
      ariaLabelledBy: 'modal-basic-title',
      size: 'medium',
      centered: true,
      scrollable: true,
      fullscreen: isMobile
    })
    .result.then((result) => {}, (reason) => {
      if(reason != 'cancel'){
        this.uploadFlag = true;
        this.loadingmsg = 'Analyzing...';
        this.progress = '0%';
        this.percentCount = 0;
        this.loaderCounter()
        this.chatExistingFileProcess(reason)
      }
    });
  }

  ngOnDestroy(): void {
    this.unlisten();
    this.citationSubjectSubscription.unsubscribe();
  }
}
