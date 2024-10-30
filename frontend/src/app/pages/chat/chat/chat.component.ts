import { AfterViewInit, Component, ElementRef, OnDestroy, OnInit, Renderer2, ViewChild } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { StreamingService } from '../services/streaming.service';
import { ActivatedRoute, Router } from '@angular/router';
import { makeElementDraggable, makeElementZoomable } from '../utils';
import { catchError, concat, of} from 'rxjs';
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { VideoWidgetComponent } from '../video-widget/video-widget.component'
import { HeaderService } from 'src/app/common/services/header.service';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('messageListBox', { read: ElementRef }) messageListBox!: ElementRef;
  objId: any;
  title: string = '';
  messageList: any = [];
  queryList: any = [];
  showBottomIcon: boolean = false;
  isMobile = window.innerWidth < 1024;
  querySent: boolean = false;
  errorInQuery: boolean = false;
  query: string = '';
  isCitationWindowOpen: boolean = false;
  isVideoCitationWindowOpen: boolean = false;
  videoCitationModalRef: any
  videoCitationItem: any
  citationLink: any;
  sourceLink: any;
  zoomableEl: any;
  draggableEl: any;
  isZoomed: boolean = false;
  errorInLoading: boolean = false;
  regenrateQuery: boolean = false;
  resize_el: any;
  m_pos: any
  chat_flow: string = '';
  redirectFrom: any;
  previewData: any[] = [];
  navigatorsubscriber: any;
  queryInitiate: any;
  exportInitiate: any;
  citationInitiate: any;
  subscription: any
  unlisten = () => {};

  constructor (
    private chatService: ChatService,
    private router: Router,
    private activatedRoute: ActivatedRoute,
    private modalService: NgbModal,
    private streamingService: StreamingService,
    private elementRef: ElementRef,
    private renderer: Renderer2,
    private headerService: HeaderService
  ) {
    this.activatedRoute.params.subscribe(params => {
      if (typeof params != undefined) {
        this.objId = JSON.parse(atob(decodeURIComponent(params?.['id'])))
        this.chat_flow = JSON.parse(atob(decodeURIComponent(params?.['type'])))
        if (this.querySent != true && this.objId != 'new' && localStorage.getItem('nav') == null) {
          this.messageList = [];
          this.queryList = [];
          this.showBottomIcon = false;
          this.closeCitationWindow()
          this.getChatMessages()
        } else {
          localStorage.removeItem('nav')
        }
      }
    });
  }

  ngOnInit(): void {
    this.queryInitiate = this.chatService.querySent.subscribe((res: any)=>{
      this.querySent = res;
      this.objId =='new' && this.querySent == true ? this.queryData(): '';
    })

    this.exportInitiate = this.headerService.isExportSubject.subscribe(param => {
      if (param) {
        this.saveAsTextFile()
      }
    })

    this.citationInitiate = this.headerService.citationSubject.subscribe(param => {
      this.isCitationWindowOpen = param;
    })

    this.setHeader()
  }

  ngAfterViewInit(): void {
    this.addClickListener()
  }

  setHeader = () => {
    // Set TRUE for mobile and above, FALSE for laptop and above
    this.headerService.setIsMobileHeaderVisibleSubject(window.innerWidth < 1024)
  }

  addClickListener = () => {
    const mobileWrapper = this.elementRef.nativeElement.querySelector('.mobile-wrapper')

    if (mobileWrapper) {
      this.unlisten = this.renderer.listen(mobileWrapper, 'click', () => {
        this.handleClick()
      })
    }
  }

  handleClick = () => {
    this.headerService.setSidebarVisible(false)
  }

  ngOnDestroy(){
    if (typeof this.subscription != 'undefined') {this.subscription.unsubscribe()}
    if (typeof this.queryInitiate != 'undefined') {this.queryInitiate.unsubscribe()}
    if (typeof this.exportInitiate != 'undefined') {this.exportInitiate.unsubscribe()}
    if (typeof this.citationInitiate != 'undefined') {this.citationInitiate.unsubscribe()}

    this.unlisten();
  }

  regenerateQuery() {
    if (typeof this.queryInitiate != 'undefined') {
    this.queryInitiate.unsubscribe()
    }
    this.regenrateQuery = true;
    this.queryData();
  }

  openCitationWindow(item: any) {
    if (this.chat_flow == 'general') {
      if (item.citation_type == 'kb') {
        this.isCitationWindowOpen = true;
        this.citationLink = item.image_path;
        this.sourceLink = item.source_path;
        if (window.innerWidth >= 1024) {
          // for desktop
          this.resetZoom();
          setTimeout(() => {
            this.draggableEl = makeElementDraggable({
              element: document.querySelector('#picture') as HTMLElement,
              isOverflowHiddenDesired: true,
            });
  
            this.zoomableEl = makeElementZoomable({
              element: document.querySelector('#picture') as HTMLElement,
              steps: 40,
            });
            this.isZoomed = true;
            this.chatService.dragElement(document.getElementById("separator") as HTMLElement, "H");
          }, 100);
        }
      } else if (item.citation_type == 'video') {
        this.openVideoCitationModal(item)
      } else {
        this.isCitationWindowOpen = false;
        this.citationLink = '';
        let url: any = item.url
        window.open(url, '_blank');
      }
    } else {
      if (window.innerWidth >= 1024) {
        // for desktop
        this.scrollAndHighlight(item)
      } else {
        // for mobile
        this.headerService.setCitationSubject(true)
        setTimeout(() => {
          this.scrollAndHighlight(item)
        }, 400)
      }
    }
  }

  scrollAndHighlight = (item: any) => {
    const elements = document.querySelectorAll(".page");
    elements.forEach((el: any) => {
      el.style.border = "none";
    });
    const element = document.querySelector("#page_" + item?.citation_num) as HTMLElement;
    if (typeof element != 'undefined') {
      if (window.innerWidth < 1024) {
        // for MOBILE and above
        element.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" })
      } else {
        // for LAPTOP and above
        element.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" })
      }
      element.style.border = "2px solid orange";
    }
  }

  openVideoCitationModal(item: any) {
    this.isVideoCitationWindowOpen = true
    this.videoCitationModalRef = this.modalService.open(VideoWidgetComponent, { backdrop: 'static', ariaLabelledBy: 'modal-basic-title', size: 'medium', centered: true, scrollable: true })
    this.videoCitationModalRef.componentInstance.item = item
    this.videoCitationModalRef.componentInstance.refreshChatwindow.subscribe((res: any) => {
      this.videoCitationItem = res
      this.getChatMessages();
    })
    this.videoCitationModalRef.result.then((result: any) => { }, (reason: any) => {
      this.isVideoCitationWindowOpen = false
    });
  }

  initZoom() {
    if (!this.isZoomed) {
      setTimeout(() => {
        this.draggableEl = makeElementDraggable({
          element: document.querySelector('#picture') as HTMLElement,
          isOverflowHiddenDesired: true,
        });

        this.zoomableEl = makeElementZoomable({
          element: document.querySelector('#picture') as HTMLElement,
          steps: 40,
        });
        this.isZoomed = true;
      }, 100);
    }
  }

  resetZoom() {
    if (this.isZoomed) {
      this.zoomableEl.stopZoomableBehavior();
      this.draggableEl.stopDraggableBehavior();
      this.isZoomed = false;
      this.initZoom()
    }
  }

  zoomInpdf() {
    const element = document.querySelectorAll(".page");
    element.forEach((el: any) => {
      el.style.width = (el.clientWidth * 0.9) + "px";
    });
  }

  zoomOutpdf() {
    const element = document.querySelectorAll(".page");
    element.forEach((el: any) => {
      el.style.width = (el.clientWidth / 0.9) >= 820 ? "820px" : (el.clientWidth / 0.9) + "px";
    });
  }

  resetpdfZoom() {
    const element = document.querySelectorAll(".page");
    element.forEach((el: any) => {
      el.style.width = '100%';
    });
  }

  closeCitationWindow() {
    this.isCitationWindowOpen = false;
    this.citationLink = '';
  }

  queryListUpdated(data: any, role?: any) {
    this.queryList.push(data);
    if (role == 'user') {
      setTimeout(() => {
        this.scrollBottom();
      }, 100);
    } else {
      this.scrollToLastChild(".sent");
    }
  }

  async queryData() {
    this.chatService.query.subscribe((res: any) => {
      this.chatService.querySent.subscribe((res: boolean) => {
        this.querySent = res;
      })
      this.query = res;
    })
    if (this.query != '' && this.querySent == true && !this.regenrateQuery) {
      this.queryListUpdated({ "role": "user", "data": [{ "content": this.query }], "citations": [] }, 'user')
    }
    this.querySent = this.regenrateQuery? true :this.querySent;
    let formData = {
      query: this.query,
      chat_id: this.objId != 'new' ? this.objId : ''
    }
    this.chatService.stream.next(true)
    let lastIndex = this.queryList.length
    const reader = await this.streamingService.chatQueryStreaming(formData)

    // Create a new TextDecoder to decode the streamed response text
    const decoder = new TextDecoder();
    let chunks = "";
    let thead = 0
    let theadtext = ""
    let isTable = false
    let tbody = 0
    let trow = ''
    let isthead = 0
    let istrow = 0
    if (reader) {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) { break; }
          // Process the streamed data
          const decodechunks = decoder.decode(value);
          const lines = decodechunks.split("\n\n")
          const parseLines = lines.map(line => line.replace("data:", "").trim())
            .filter((line) => line !== "")
            .map((line) => {
              return JSON.parse(line)
            })

          parseLines.forEach((parseLine: any) => {
            if (typeof parseLine.text != 'undefined') {
              this.querySent = false;
              let paresStr = parseLine.text.replace(/[]/g, "").replace('>\n','').replace('</','');
              if(isTable){
                if(paresStr == "<thead"){
                  chunks += "<thead>";
                  thead = 1
                }else if(thead == 1){
                  if(paresStr == "thead"){
                    chunks += "</thead>"
                    thead = 0;
                  }else{
                    theadtext = parseLine.text.replace(/[]/g, "").replace('>\n','').replace('</','').replace(/[<>]/g, "")
                    if(theadtext == "|"){
                      isthead = 1
                      chunks += "<tr><th scope='col'>";
                    }else if(isthead == 1 && theadtext == " |")
                    {
                      chunks += "</th><th scope='col'>";
                    }else if(isthead == 1 && theadtext == " |\n"){
                      chunks += "</th></tr>";
                      isthead = 0
                    }
                    else{
                      chunks += theadtext.replace('\n','');
                    }
                  }
                }else if(paresStr == "<tbody"){
                  chunks += "<tbody>";
                  tbody = 1
                }else if(tbody == 1){
                  if(paresStr == "tbody"){
                    chunks += "</tbody>"
                    tbody = 0;
                  }else{
                    trow = parseLine.text.replace(/[]/g, "").replace('>\n','').replace('</','').replace(/[<>]/g, "")
                    if(trow == "|"){
                      istrow = 1
                      chunks += "<tr><td scope='row'>";
                    }else if(istrow == 1 && trow == " |")
                    {
                      chunks += "</td><td scope='row'>";
                    }else if(istrow == 1 && trow == " |\n"){
                      chunks += "</td></tr>";
                      istrow = 0
                    }
                    else{
                      chunks += trow.replace('\n','');
                    }
                  }
                }else if(paresStr == "table"){
                  chunks += "</table></div>"
                  isTable = false
                }
              }else{
                if(paresStr == "<table"){
                  chunks += "<div class='table-responsive'><table class='table table-hover table-striped'>"
                  isTable = true
                }else{
                  chunks += parseLine.text.replace('>\n','').replace('</','').replace(/[<>]/g, "");
                }
              }
              
              this.queryList[lastIndex] = { "role": "assistant", "data": [{ "content": chunks }], "citations": [], "keywords": [], "sample_questions": [] };
              if(this.isUserNearBottom()){
                setTimeout(() => {
                  this.scrollBottom();
                }, 100);
              }
            } else {
              this.chatService.getChatMessage({ "chat_id": parseLine?._id }).subscribe((res: any) => {
                this.chatService.stream.next(false)
                console.log(res)
                this.chatService.querySent.next(false);
                this.querySent = false;
                this.regenrateQuery = false;

                this.queryList[lastIndex] = res?.chat_conversations[0]
                if(this.isUserNearBottom()){
                  setTimeout(() => {
                    this.scrollBottom();
                  }, 100);
                }

                if (this.objId == 'new') {
                  let objId = encodeURI(btoa(JSON.stringify(parseLine?._id)).toString())
                  let type = encodeURI(btoa(JSON.stringify(res?.chat_flow)).toString())
                  localStorage.setItem('nav', '1')
                  this.router.navigate(['/chat/c/', objId, type]);
                  this.chatService.newQuery.next({ '_id': parseLine?._id, 'title': res?.title, 'updated_on': res?.updated_on, 'chat_flow': res?.chat_flow });
                }
              },
                (err) => {
                  this.chatService.stream.next(false)
                })
            }
          })
        }
      } catch (error: any) {
        this.chatService.stream.next(false)
        this.chatService.querySent.next(false);
        this.querySent = false;
        this.regenrateQuery = false;
        this.queryListUpdated({ "role": "assistant", "data": [{ "content": "Looks like something went wrong." }], "citations": [], "keywords": [], "sample_questions": [], "error": true }, 'assistant')
      }
    } else {
      this.chatService.stream.next(false)
      this.chatService.querySent.next(false);
      this.querySent = false;
      this.regenrateQuery = false;
      this.queryListUpdated({ "role": "assistant", "data": [{ "content": "Looks like something went wrong." }], "citations": [], "keywords": [], "sample_questions": [], "error": true }, 'assistant')
    }
  }

  getChatMessages() {
    let resArr: any[] = [];
    if (typeof this.subscription != 'undefined') {
      this.subscription.unsubscribe();
    }
    this.chat_flow != 'general' ?
    this.subscription = concat(
      this.chatService.getChatMessages({ "chat_id": this.objId }).pipe(catchError(err => of({isError: true, error: err})))
    , this.chatService.chatPreview({ "chat_id": this.objId }).pipe(catchError(err => of({isError: true, error: err})))
    ).subscribe((res: any) => {
        this.errorInLoading = false;
          resArr.push(res);
          if(resArr.length > 0){
            if(!resArr[0].hasOwnProperty('isError')){
              this.setItems(resArr[0])
            }else{
            }
          }else{
           
          }
          if(resArr.length > 1){
            this.previewData = !resArr[1].hasOwnProperty('isError')? resArr[1].file_preview: [];
          }

        }, err => {
          console.log(err)
          this.errorInLoading = true;
          // hide preview and export icon if error
          this.headerService.setIsPreviewIconVisible(false)
          this.headerService.setIsExportIconVisible(false)
        })
      :
    this.subscription = this.chatService.getChatMessages({ "chat_id": this.objId }) 
    .subscribe((res: any) => {
        this.setItems(res)
      },
        (err) => {
          this.errorInLoading = true;
          // hide preview and export icon if error
          this.headerService.setIsPreviewIconVisible(false)
          this.headerService.setIsExportIconVisible(false)
        })
  }

  setItems(data: any) {
    this.errorInLoading = false;
    this.messageList = data.chat_conversations;
    if (this.isVideoCitationWindowOpen) { // for video citation link refresh
      let newCitationItem = this.messageList[this.videoCitationItem?.parentIndex].citations.find((d: any) => {
        return d.citation_num == this.videoCitationItem.citation_num
      })
      newCitationItem.parentIndex = this.videoCitationItem.parentIndex
      this.videoCitationModalRef.close()
      this.openVideoCitationModal(newCitationItem)
    }

    this.title = data.title;
    this.chat_flow = data.chat_flow;

    if (this.chat_flow != 'general') {
      this.sourceLink = data.source_path;
      if (window.innerWidth >= 1024) {
        // for desktop
        this.isCitationWindowOpen = true;
      }
      this.headerService.setIsPreviewIconVisible(true)
      setTimeout(() => {// enable slider
        const separatorEl = document.getElementById("separator") as HTMLElement

        if (separatorEl) {
          this.chatService.dragElement(separatorEl, "H" );
        }
      } , 300);
    }else{
      this.isCitationWindowOpen = false;
      this.headerService.setIsPreviewIconVisible(false)
    }
    setTimeout(() => {
      if (data.chat_flow != 'general' && typeof (this.messageList?.[1]) == 'undefined') {
      }
      else if (data.chat_flow == 'general' && typeof (this.messageList?.[2]) == 'undefined') {
        this.scrollToLastChild(".sent");
      }
      else {
        this.scrollBottom();
      }
    }, 100);
  }

  scrollBottom() {
    this.messageListBox.nativeElement.scrollTo({ top: (this.messageListBox.nativeElement.scrollHeight), behavior: 'auto' });
  }

  scrollToLastChild(component: any) {
    setTimeout(() => {
      const element = document.querySelectorAll(component);
      typeof element != 'undefined' && element.length != 0 ? element[element.length - 1].scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" }) : '';
    }, 300);
  }

  private isUserNearBottom(): boolean {
    const threshold = 120;
    const position = this.messageListBox.nativeElement.scrollTop + this.messageListBox.nativeElement.offsetHeight;
    const height = this.messageListBox.nativeElement.scrollHeight;
    return position > height - threshold;
  }

  checkEnd(e: any) {
    let scrollHeight = this.messageListBox.nativeElement.scrollHeight - this.messageListBox.nativeElement.clientHeight;
    if (scrollHeight == this.messageListBox.nativeElement.scrollTop) {
      this.showBottomIcon = false;
    } else {
      this.showBottomIcon = true;
    }
  }

  saveAsTextFile() {
    let txtData = '';
    this.messageList.filter((item: any) => {

      item.summary != '' ? txtData += "AIXponent: " + item.summary : '';//add Summary

      if (item.keywords.length > 0) { //add Key Topics
        txtData += '\n\n';
        txtData += 'Key Topics: ';
        txtData += item.keywords.map((data: any) => {
          return ' ' + data;
        })
      }

      if (item.sample_questions.length > 0) { //add Follow-up Questions
        txtData += '\n\n';
        txtData += 'Consider these inquiries below that may arise\t';
        item.sample_questions.map((data: any, index: any) => {
          txtData += '\n\n' + (index + 1) + '. ' + data;
        })
      }

      item.summary == '' ? txtData += item.role == 'user' ? '\n\nME: ' : '\n\nAIXponent: ' : '';
      item.data.map((data: any) => { // add chat messages
        txtData += data.content != "" ? data.content : '';
      })
    })

    this.queryList.filter((item: any) => {
      txtData += item.role == 'user' ? '\n\nME: ' : '\n\nAIXponent: ';
      item.data.map((data: any) => { // add chat messages
        txtData += data.content != "" ? data.content : '';
      })
    })

    this.chatService.exportAsTxt(txtData, this.title);
  }

  openFile(file_path: any) {
    window.open(file_path, '_blank')
  }

  hideCitationPanel = () => {
    this.isCitationWindowOpen = false;
  }
}
