import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { CardModule } from 'primeng/card';
import { InputTextModule } from 'primeng/inputtext';
import { InputTextareaModule } from 'primeng/inputtextarea';
import { MenuModule } from 'primeng/menu';
import { MenubarModule } from 'primeng/menubar';
import { PanelMenuModule } from 'primeng/panelmenu';
import { ButtonModule } from 'primeng/button';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TableModule } from 'primeng/table';
import { ChipsModule } from 'primeng/chips';
import { ChipModule } from 'primeng/chip';
import { AccordionModule } from 'primeng/accordion';
import { SafePipe } from '../common/pipe/safe.pipe';
import { Draggable } from '../common/directives/draggable-img';
import { BreadcrumbModule } from 'primeng/breadcrumb';
import { ConfirmPopupModule } from 'primeng/confirmpopup';
import { DialogModule } from 'primeng/dialog';
import { ToastModule } from 'primeng/toast';
import { DragDropModule } from 'primeng/dragdrop';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { GalleriaModule } from 'primeng/galleria';
import { ProgressBarModule } from 'primeng/progressbar';
import { SkeletonModule } from 'primeng/skeleton';
import { TooltipModule } from 'primeng/tooltip';
import { AutoCompleteModule } from 'primeng/autocomplete';
import { SelectButtonModule } from 'primeng/selectbutton';
import { ToggleButtonModule } from 'primeng/togglebutton';
import { CheckboxModule } from 'primeng/checkbox';
import { OverlayPanelModule } from 'primeng/overlaypanel';
import { ImageModule } from 'primeng/image';
import { SidebarModule } from 'primeng/sidebar';
import { DropdownModule } from 'primeng/dropdown';
import { MultiSelectModule } from 'primeng/multiselect';
import { TreeModule } from 'primeng/tree';
import { MessagesModule } from 'primeng/messages';
import { ClipboardModule } from '@angular/cdk/clipboard';
import { TagModule } from 'primeng/tag';
import { PanelModule } from 'primeng/panel';
import { InputSwitchModule } from 'primeng/inputswitch';

@NgModule({
  declarations: [SafePipe, Draggable],
  imports: [
    CommonModule,
    FormsModule, 
    ReactiveFormsModule,
    CardModule,
    InputTextModule,
    InputTextareaModule,
    MenubarModule,
    MenuModule,
    PanelMenuModule, 
    ButtonModule,
    ProgressSpinnerModule,
    TableModule,
    ChipsModule,
    ChipModule,
    BreadcrumbModule,
    ConfirmPopupModule,
    DialogModule,
    ToastModule,
    DragDropModule,
    ConfirmDialogModule,
    AccordionModule,
    GalleriaModule,
    ProgressBarModule,
    SkeletonModule,
    TooltipModule,
    AutoCompleteModule,
    SelectButtonModule,
    ToggleButtonModule,
    CheckboxModule,
    OverlayPanelModule,
    ImageModule,
    NgbModule,
    SidebarModule,
    DropdownModule,
    MultiSelectModule,
    TreeModule,
    MessagesModule,
    ClipboardModule,
    TagModule,
    PanelModule,
    InputSwitchModule
  ],
  exports:[
    CardModule,
    FormsModule, 
    ReactiveFormsModule,
    InputTextModule,
    InputTextareaModule,
    MenubarModule,
    MenuModule,
    PanelMenuModule,
    ButtonModule,
    ProgressSpinnerModule,
    TableModule,
    ChipsModule,
    ChipModule,
    BreadcrumbModule,
    ConfirmPopupModule,
    DialogModule,
    AccordionModule,
    ToastModule,
    DragDropModule,
    ConfirmDialogModule,
    GalleriaModule,
    ProgressBarModule,
    SkeletonModule,
    TooltipModule,
    AutoCompleteModule,
    SelectButtonModule,
    ToggleButtonModule,
    CheckboxModule,
    OverlayPanelModule,
    SafePipe,
    Draggable,
    ImageModule,
    NgbModule,
    SidebarModule,
    DropdownModule,
    MultiSelectModule,
    TreeModule,
    MessagesModule,
    ClipboardModule,
    TagModule,
    PanelModule,
    InputSwitchModule
  ]
})

export class SharedModule { }
