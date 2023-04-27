package hse.project.sipserviceauth.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UpdateResponse<T> {

    private String message;

    private String[] updatedFields;

    private T updatedObject;
}
