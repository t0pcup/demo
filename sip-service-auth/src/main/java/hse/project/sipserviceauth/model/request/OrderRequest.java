package hse.project.sipserviceauth.model.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class OrderRequest {

    private String url;

    private String name;

    private String model;

    private String satellite;

    private String url2;
}
