import { Component, ElementRef, ViewChild } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { StreamingService } from '../services/streaming.service';
import { ActivatedRoute, Router } from '@angular/router';
import { forkJoin } from 'rxjs';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent {
  @ViewChild('messageListBox', { read: ElementRef }) messageListBox!: ElementRef;
  objId: any = 'new';
  title: string = '';
  messageList: any = [];
  queryList: any = [];
  showBottomIcon: boolean = false;
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
  previewData: any;
  navigatorsubscriber: any;
  queryInitiate: any;
  subscription: any
  graphsubscriber: any
  constructor(private chatService: ChatService,
    private router: Router,
    private activatedRoute: ActivatedRoute,
    private streamingService: StreamingService,
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
      this.objId =='new' && res == true ? this.queryData(): '';
    })
  }

  ngOnDestroy(){
    if (typeof this.subscription != 'undefined') {this.subscription.unsubscribe()}
    if (typeof this.queryInitiate != 'undefined') {this.queryInitiate.unsubscribe()}
  }

  regenerateQuery() {
    if (typeof this.queryInitiate != 'undefined') {
    this.queryInitiate.unsubscribe()
    }
    this.regenrateQuery = true;
    this.queryData();
  }

  openCitationWindow(item: any) {
  }

  openVideoCitationModal(item: any) {
  }

  initZoom() {
  }

  resetZoom() {
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

  queryListUpdated({ data, role, lastIndex }: { data: any; role?: any; lastIndex: any; }) {
    this.queryList[lastIndex] = data;
    if (role == 'user') {
      setTimeout(() => {
        this.scrollBottom();
      }, 100);
    } else {
      this.scrollToLastChild(".sent");
    }
  }

  saveFeedback(data: any) {
    let formData = {
      chat_id: this.objId,
      index: data.index,
      feedback: data.type
    }
    this.chatService.saveFeedBack(formData).subscribe((res: any) => {

    },
      (err) => {

      })
  }

  async queryData() {
    this.chatService.query.subscribe((res: any) => {
      this.chatService.querySent.subscribe((res: boolean) => {
        this.querySent = res;
      })
      this.query = res;
    })
    if (this.query != '' && this.querySent == true && !this.regenrateQuery) {
      let ulastIndex = this.queryList.length; 
      this.queryListUpdated({ data: { "role": "user", "data": { "content": this.query }, "citations": [] }, role: 'user', lastIndex: ulastIndex })
    }
    this.querySent = this.regenrateQuery? true :this.querySent;

    if(this.query != ''){
      this.chatService.stream.next(true)
      let lastIndex = this.queryList.length;  
      let formData: any = {
        query: this.query
      }
      
      this.objId != 'new' ? formData.chat_id = this.objId : '';

      this.chatService.getSDAType(formData).subscribe({
        next: (typeItem: any) =>{
          this.chatService.querySent.next(false);
          this.querySent = false;   
          this.queryList[lastIndex] = { 
            "role": "assistant", 
            "data": {sql_query: typeItem.sql_query}, 
            "citations": [], 
            "keywords": [], 
            "feedback":2,
            "sample_questions": [], 
            "loader": 1 ,
            "grapherror": false
          };
          if(this.isUserNearBottom()){
            setTimeout(() => {
              this.scrollBottom();
            }, 100);
          }

          let payload  = {
            chat_id: typeItem.chat_id,
            file_location: typeItem.csv_data,
            graph_type: typeItem.graph_type,
            query: typeItem.user_query,
          }

          forkJoin([this.onlyText(lastIndex, payload),this.onlyGraph(lastIndex, payload)])

          if (this.objId == 'new') {
            this.chatService.newQuery.next({ '_id': typeItem?.chat_id, 'title': typeItem?.title, 'updated_on': typeItem?.updated_on, 'chat_flow': typeItem?.chat_flow });
            let objId = encodeURI(btoa(JSON.stringify(typeItem.chat_id)).toString())
            let type = encodeURI(btoa(JSON.stringify(typeItem?.chat_flow)).toString())
            localStorage.setItem('nav', '1')
            this.router.navigate(['/structure-data-analysis/c/', objId, type]);
          }

        },
        error:(typeItemError: any) =>{
          this.chatService.stream.next(false)
          this.chatService.querySent.next(false);
          this.querySent = false;
          this.regenrateQuery = false;
          this.queryListUpdated({ data: { "role": "assistant", "data": { "content": "Looks like something went wrong." }, "citations": [], "keywords": [], "sample_questions": [], "error": true }, role: 'assistant', lastIndex })
    
        }
      })
    }
  }


  onlyGraph(lastIndex: any, payload: any){
    this.graphsubscriber =  this.chatService.getSDAVisualization(payload).subscribe({
      next: (typeItem: any) =>{
        this.queryList[lastIndex].data.graphjson = typeItem;
        this.queryList[lastIndex].loader = 3;
        if(this.isUserNearBottom()){
          this.scrollToLastChild('.visualization')
        }
      },
      error:(typeItemError: any) =>{
        setTimeout(() => {
          this.queryList[lastIndex].loader = 3;
          this.queryList[lastIndex].grapherror = true;
        }, 100);
        
      }
    })
  }

  async onlyText(lastIndex: any,payload: any){
    const reader = await this.streamingService.chatQueryStreaming(payload)
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
          if (done) { 
            if(!('graphjson' in this.queryList[lastIndex].data) && !this.queryList[lastIndex].grapherror){
              this.queryList[lastIndex].loader = 2
            }
            break; 
          }
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
              this.queryList[lastIndex].data.content = chunks;
              this.queryList[lastIndex].loader = 3
              if(this.isUserNearBottom()){
                setTimeout(() => {
                  this.scrollBottom();
                }, 100);
              }
            }
          })
        }
      } catch (error: any) {
        this.graphsubscriber.unsubscribe()
        this.queryList[lastIndex].loader = 3
        this.chatService.stream.next(false)
        this.chatService.querySent.next(false);
        this.querySent = false;
        this.regenrateQuery = false;
        this.queryListUpdated({ data: { "role": "assistant", "data": { "content": "Looks like something went wrong." }, "citations": [], "keywords": [], "sample_questions": [], "error": true }, role: 'assistant', lastIndex })
      }
    } else {
      this.graphsubscriber.unsubscribe()
      this.queryList[lastIndex].loader = 3
      this.chatService.stream.next(false)
      this.chatService.querySent.next(false);
      this.querySent = false;
      this.regenrateQuery = false;
      this.queryListUpdated({ data: { "role": "assistant", "data": { "content": "Looks like something went wrong." }, "citations": [], "keywords": [], "sample_questions": [], "error": true }, role: 'assistant', lastIndex })
    }
  }

  getChatMessages() {
    if (typeof this.subscription != 'undefined') {
      this.subscription.unsubscribe();
    }
    this.subscription = this.chatService.getChatMessages({ "chat_id": this.objId }) 
    .subscribe((res: any) => {
        this.setItems(res)
      },
        (err) => {
          this.errorInLoading = true;
        })
  }

  setItems(data: any) {
    this.errorInLoading = false;
    this.messageList = data.chat_conversations;
    if (this.isVideoCitationWindowOpen) { // for video citation link refresh
      let newCitationItem = this.messageList[this.videoCitationItem.parentIndex].citations.find((d: any) => {
        return d.citation_num == this.videoCitationItem.citation_num
      })
      newCitationItem.parentIndex = this.videoCitationItem.parentIndex
      this.videoCitationModalRef.close()
      this.openVideoCitationModal(newCitationItem)
    }

    this.title = data.title;
    this.chat_flow = data.chat_flow;

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
    const position = this.messageListBox?.nativeElement?.scrollTop + this.messageListBox?.nativeElement?.offsetHeight;
    const height = this.messageListBox?.nativeElement?.scrollHeight;
    return position > height - threshold;
  }

  checkEnd(e: any) {
    let scrollHeight = this.messageListBox?.nativeElement?.scrollHeight - this.messageListBox?.nativeElement?.clientHeight;
    if (scrollHeight == this.messageListBox?.nativeElement?.scrollTop) {
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
      txtData += item.content != "" ? item.content : '';
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
}
