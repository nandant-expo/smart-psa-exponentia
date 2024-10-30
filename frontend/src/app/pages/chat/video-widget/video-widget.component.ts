import { Component, EventEmitter, Input, Output, ViewChild } from '@angular/core';
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';

@Component({
  selector: 'app-video-widget',
  templateUrl: './video-widget.component.html',
  styleUrls: ['./video-widget.component.scss']
})
export class VideoWidgetComponent {
  @ViewChild('citationvideo',{static: false}) citationvideo!: HTMLElement
  @Input() item!: any;
  notplayable = false
  loader = false
  lazyContainer: any
  @Output() refreshChatwindow = new EventEmitter<any>();
  vidURL:  any
  videoElement: any
  constructor(public activeModal: NgbActiveModal){}


  ngOnInit(): void { 
    this.lazyContainer = document.querySelector("div#lazy_container");
    this.videoDynamicRendering()
  }

  videoDynamicRendering(){
    console.log("called")

    this.videoElement = document.querySelector("video");
    this.vidURL = this.item?.url
    this.videoElement.load()
    this.videoElement.addEventListener('error', (event: any) => {
      if(event.type == 'error'){
        this.notplayable = true
        this.videoElement.remove()
      }
    });

    
    this.videoElement.onloadstart = ()=>{
      //video responsive width and height
      let width = this.videoElement.parentNode?.parentElement?.clientWidth as number - 50
      let height = this.videoElement.parentNode?.parentElement?.clientHeight as number

      this.videoElement.width = width;
      this.videoElement.height = window.innerHeight as number - height

      this.videoElement.currentTime = this.item.duration;
      this.videoElement.controls = true;
      
    } 

    this.videoElement.addEventListener("loadeddata", () => {
      this.videoElement.play()
    });
  
  }

  refreshChat(){
    this.loader = true
    this.refreshChatwindow.emit(this.item)
  }

  closeModal(){
    this.videoElement.remove()
    this.activeModal.close()
  }
}
